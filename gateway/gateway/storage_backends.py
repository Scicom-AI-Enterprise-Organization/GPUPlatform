"""Storage backends for the self-hosted HuggingFace catalog/mirror.

A thin, **synchronous** read/write interface over three backends — S3 (and
S3-compatible: R2/MinIO), a local filesystem path, and SFTP — so the HF mirror
(`hf_mirror_api.py`) and the catalog API (`catalog_api.py`) can serve and store
repo files without caring where the bytes live.

All methods are blocking; callers in the async gateway must invoke them via
`fastapi.concurrency.run_in_threadpool` (or `asyncio.to_thread`).

Keys are **relative to the storage's own root** (its `config.prefix` for s3,
`config.path` for local, `config.base_path` for sftp). The catalog passes keys
already namespaced by the repo, e.g. `catalog/<ns>/<name>/config.json` or
`catalog/<ns>/<name>/.hf-lfs/<oid>`.
"""
from __future__ import annotations

import io
import json
import logging
import os
import posixpath
import shutil
from typing import Optional

from . import bench
from . import crypto
from .db import Storage

logger = logging.getLogger("gateway.storage_backends")

# Chunk size for streaming reads/writes (8 MiB).
CHUNK = 8 * 1024 * 1024


class StorageError(Exception):
    """Raised for backend misconfiguration or unreachable targets."""


# ---------- key helpers -------------------------------------------------


def _norm_key(key: str) -> str:
    """Collapse, strip leading slashes; reject path traversal."""
    parts = []
    for seg in key.replace("\\", "/").split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            raise StorageError(f"illegal path segment in key: {key!r}")
        parts.append(seg)
    return "/".join(parts)


def _join(base: str, key: str) -> str:
    base = (base or "").strip().strip("/")
    key = _norm_key(key)
    return f"{base}/{key}" if base else key


# ---------- interface ---------------------------------------------------


class StorageBackend:
    """Sync interface. Keys are relative to the backend's configured root."""

    presign_capable: bool = False

    def stat(self, key: str) -> Optional[int]:
        """Object size in bytes, or None if it doesn't exist."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        return self.stat(key) is not None

    def list_prefix(self, prefix: str) -> list[dict]:
        """Return [{key, size}] for every object under `prefix` (recursive).
        `key` is relative to the backend root (same space as stat/put)."""
        raise NotImplementedError

    def open_reader(self, key: str, start: int = 0):
        """Return a file-like object positioned at `start` exposing read(n) +
        close(). Raises FileNotFoundError/StorageError if missing."""
        raise NotImplementedError

    def put_file(self, key: str, local_path: str) -> int:
        """Upload a local file to `key`. Returns bytes written."""
        raise NotImplementedError

    def put_bytes(self, key: str, data: bytes) -> int:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release any held resources (sftp connections)."""
        return None


# ---------- S3 ----------------------------------------------------------


class S3Backend(StorageBackend):
    presign_capable = True

    def __init__(self, target: "bench.S3Target", base_prefix: str = ""):
        self._t = target
        self._base = (base_prefix or "").strip().strip("/")
        self._cli = bench._s3_client(target)

    def _k(self, key: str) -> str:
        return _join(self._base, key)

    def stat(self, key: str) -> Optional[int]:
        try:
            r = self._cli.head_object(Bucket=self._t.bucket, Key=self._k(key))
            return int(r["ContentLength"])
        except Exception:  # noqa: BLE001 — head 404/403 → treat as missing
            return None

    def list_prefix(self, prefix: str) -> list[dict]:
        full = self._k(prefix)
        objs = bench.s3_list(full + "/" if not full.endswith("/") else full, self._t)
        out: list[dict] = []
        cut = len(self._base) + 1 if self._base else 0
        for o in objs:
            rel = o["key"][cut:] if cut else o["key"]
            out.append({"key": rel, "size": o["size"]})
        return out

    def open_reader(self, key: str, start: int = 0):
        kwargs = {"Bucket": self._t.bucket, "Key": self._k(key)}
        if start:
            kwargs["Range"] = f"bytes={start}-"
        obj = self._cli.get_object(**kwargs)
        return obj["Body"]  # botocore StreamingBody: .read(n), .close()

    def put_file(self, key: str, local_path: str) -> int:
        # upload_file does managed multipart automatically (>5 GB safe).
        self._cli.upload_file(local_path, self._t.bucket, self._k(key))
        return os.path.getsize(local_path)

    def put_bytes(self, key: str, data: bytes) -> int:
        self._cli.put_object(Bucket=self._t.bucket, Key=self._k(key), Body=data)
        return len(data)

    def delete(self, key: str) -> None:
        self._cli.delete_object(Bucket=self._t.bucket, Key=self._k(key))


def _s3_target_for_catalog(row: Storage) -> "bench.S3Target":
    """Build an S3Target from a kind=s3 Storage row WITHOUT the benchmark-specific
    prefix_root that bench._target_from_storage_row bakes in (catalog manages its
    own key namespace)."""
    cfg = row.config or {}
    # Same precedence as benchmarks: global-secret refs > encrypted literal > env.
    access_key, secret_key = bench._resolve_s3_creds(cfg)
    return bench.S3Target(
        bucket=(cfg.get("bucket") or "").strip(),
        region=(cfg.get("region") or os.environ.get("AWS_REGION", "ap-southeast-5")),
        endpoint=(cfg.get("endpoint") or None),
        access_key=access_key,
        secret_key=secret_key,
        prefix_root="",
    )


# ---------- Local filesystem -------------------------------------------


class LocalBackend(StorageBackend):
    def __init__(self, root: str):
        if not root:
            raise StorageError("local storage has no `path` configured")
        self._root = os.path.abspath(os.path.expanduser(root))

    def _p(self, key: str) -> str:
        # _norm_key already rejects traversal; final path must stay under root.
        p = os.path.join(self._root, _norm_key(key))
        if os.path.commonpath([self._root, os.path.abspath(p)]) != self._root:
            raise StorageError(f"path escapes storage root: {key!r}")
        return p

    def stat(self, key: str) -> Optional[int]:
        try:
            return os.path.getsize(self._p(key))
        except OSError:
            return None

    def list_prefix(self, prefix: str) -> list[dict]:
        base = self._p(prefix) if prefix else self._root
        out: list[dict] = []
        if not os.path.isdir(base):
            # `prefix` may point at a file, or nothing
            if os.path.isfile(base):
                rel = os.path.relpath(base, self._root).replace(os.sep, "/")
                out.append({"key": rel, "size": os.path.getsize(base)})
            return out
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                fp = os.path.join(dirpath, fn)
                rel = os.path.relpath(fp, self._root).replace(os.sep, "/")
                try:
                    out.append({"key": rel, "size": os.path.getsize(fp)})
                except OSError:
                    continue
        return out

    def open_reader(self, key: str, start: int = 0):
        f = open(self._p(key), "rb")
        if start:
            f.seek(start)
        return f

    def put_file(self, key: str, local_path: str) -> int:
        dst = self._p(key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(local_path, tmp)
        os.replace(tmp, dst)
        return os.path.getsize(dst)

    def put_bytes(self, key: str, data: bytes) -> int:
        dst = self._p(key)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
        return len(data)

    def delete(self, key: str) -> None:
        try:
            os.remove(self._p(key))
        except FileNotFoundError:
            pass


# ---------- SFTP --------------------------------------------------------


class _SFTPReader:
    """A read()-able that owns its own SFTP connection, closed on .close().
    Each reader holds a dedicated connection so concurrent downloads never share
    a paramiko channel (which is not safe for concurrent reads)."""

    def __init__(self, transport, sftp, f):
        self._transport = transport
        self._sftp = sftp
        self._f = f

    def read(self, n: int) -> bytes:
        return self._f.read(n)

    def close(self) -> None:
        for c in (self._f, self._sftp, self._transport):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass


class SFTPBackend(StorageBackend):
    """Connection-per-operation SFTP backend. Each call opens a fresh transport,
    runs its op, and closes it (a streaming read keeps its connection alive via
    `_SFTPReader` until the consumer closes it). No shared client → thread-safe
    under concurrent downloads, at the cost of a connect per op."""

    def __init__(self, cfg: dict):
        import paramiko  # lazy — only needed for sftp storages

        self._paramiko = paramiko
        self._host = (cfg.get("host") or "").strip()
        self._port = int(cfg.get("port") or 22)
        self._user = (cfg.get("username") or "").strip()
        # Keep a leading slash so an absolute base_path stays absolute; SFTP paths
        # without a leading slash are relative to the login home dir.
        self._base = (cfg.get("base_path") or "").strip().rstrip("/")
        if not self._host or not self._user:
            raise StorageError("sftp storage needs host + username")
        enc = cfg.get("credentials_enc")
        creds = json.loads(crypto.decrypt(enc)) if enc else {}
        self._password = creds.get("password")
        self._private_key = creds.get("privateKey") or creds.get("private_key")

    def _connect(self):
        """Return a fresh (transport, sftp). Caller must close the transport."""
        p = self._paramiko
        transport = p.Transport((self._host, self._port))
        pkey = None
        if self._private_key:
            for kc in (p.Ed25519Key, p.RSAKey, p.ECDSAKey):
                try:
                    pkey = kc.from_private_key(io.StringIO(self._private_key))
                    break
                except Exception:  # noqa: BLE001 — try next key type
                    continue
        try:
            if pkey is not None:
                transport.connect(username=self._user, pkey=pkey)
            else:
                transport.connect(username=self._user, password=self._password)
        except Exception as e:  # noqa: BLE001
            try:
                transport.close()
            except Exception:  # noqa: BLE001
                pass
            raise StorageError(f"sftp connect failed: {e}") from e
        return transport, p.SFTPClient.from_transport(transport)

    def ping(self) -> None:
        """Open a connection + stat the base path. Raises StorageError on failure."""
        transport, sftp = self._connect()
        try:
            sftp.stat(self._base or ".")
        except IOError as e:
            raise StorageError(str(e)) from e
        finally:
            transport.close()

    def _p(self, key: str) -> str:
        key = _norm_key(key)
        if not self._base:
            return key
        return f"{self._base}/{key}" if key else self._base

    def _mkdirs(self, sftp, remote_dir: str) -> None:
        if not remote_dir or remote_dir == "/":
            return
        parts = [p for p in remote_dir.split("/") if p]
        cur = "/" if remote_dir.startswith("/") else ""
        for part in parts:
            cur = f"{cur}{part}" if cur in ("", "/") else f"{cur}/{part}"
            try:
                sftp.stat(cur)
            except IOError:
                try:
                    sftp.mkdir(cur)
                except IOError:
                    pass

    def stat(self, key: str) -> Optional[int]:
        transport, sftp = self._connect()
        try:
            return int(sftp.stat(self._p(key)).st_size)
        except IOError:
            return None
        finally:
            transport.close()

    def list_prefix(self, prefix: str) -> list[dict]:
        import stat as _stat

        transport, sftp = self._connect()
        base = self._p(prefix) if prefix else self._base
        out: list[dict] = []
        cut = len(self._base) + 1 if self._base else 0

        def _walk(path: str) -> None:
            try:
                entries = sftp.listdir_attr(path)
            except IOError:
                return
            for ent in entries:
                full = posixpath.join(path, ent.filename)
                if _stat.S_ISDIR(ent.st_mode):
                    _walk(full)
                else:
                    out.append({"key": full[cut:] if cut else full, "size": int(ent.st_size)})

        try:
            try:
                st = sftp.stat(base)
            except IOError:
                return out
            if not _stat.S_ISDIR(st.st_mode):
                out.append({"key": base[cut:] if cut else base, "size": int(st.st_size)})
                return out
            _walk(base)
            return out
        finally:
            transport.close()

    def open_reader(self, key: str, start: int = 0):
        transport, sftp = self._connect()
        try:
            f = sftp.open(self._p(key), "rb")
        except IOError as e:
            transport.close()
            raise FileNotFoundError(str(e)) from e
        try:
            f.prefetch()
        except Exception:  # noqa: BLE001 — prefetch is an optimisation
            pass
        if start:
            f.seek(start)
        return _SFTPReader(transport, sftp, f)

    def put_file(self, key: str, local_path: str) -> int:
        transport, sftp = self._connect()
        try:
            remote = self._p(key)
            self._mkdirs(sftp, posixpath.dirname(remote))
            sftp.put(local_path, remote)
            return int(sftp.stat(remote).st_size)
        finally:
            transport.close()

    def put_bytes(self, key: str, data: bytes) -> int:
        transport, sftp = self._connect()
        try:
            remote = self._p(key)
            self._mkdirs(sftp, posixpath.dirname(remote))
            sftp.putfo(io.BytesIO(data), remote)
            return len(data)
        finally:
            transport.close()

    def delete(self, key: str) -> None:
        transport, sftp = self._connect()
        try:
            sftp.remove(self._p(key))
        except IOError:
            pass
        finally:
            transport.close()


# ---------- dispatch ----------------------------------------------------


def resolve_backend(row: Storage) -> StorageBackend:
    """Build the backend for a Storage row. Raises StorageError for unsupported
    or misconfigured kinds. The returned backend roots keys at the storage's own
    base location (s3 prefix / local path / sftp base_path)."""
    if row is None:
        raise StorageError("no storage configured")
    kind = row.kind
    cfg = row.config or {}
    if kind == "s3":
        target = _s3_target_for_catalog(row)
        if not target.bucket:
            raise StorageError("s3 storage has no bucket configured")
        return S3Backend(target, base_prefix=(cfg.get("prefix") or ""))
    if kind == "local":
        return LocalBackend(cfg.get("path") or "")
    if kind == "sftp":
        return SFTPBackend(cfg)
    raise StorageError(f"storage kind {kind!r} cannot host catalog repos (need s3/local/sftp)")
