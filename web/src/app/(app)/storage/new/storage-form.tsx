"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { gateway } from "@/lib/gateway";
import type { StorageKind } from "@/lib/types";

type TestState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "ok"; message: string }
  | { status: "fail"; message: string };

export function StorageForm() {
  const router = useRouter();

  const [kind, setKind] = useState<StorageKind>("s3");
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");

  // s3
  const [bucket, setBucket] = useState("");
  const [prefix, setPrefix] = useState("");
  const [region, setRegion] = useState("");
  const [endpoint, setEndpoint] = useState("");
  // s3 credentials: pasted keys, or picked from global secrets.
  const [s3CredSource, setS3CredSource] = useState<"paste" | "secret">("paste");
  const [accessKeyId, setAccessKeyId] = useState("");
  const [secretAccessKey, setSecretAccessKey] = useState("");
  const [accessKeyIdSecret, setAccessKeyIdSecret] = useState("");
  const [secretAccessKeySecret, setSecretAccessKeySecret] = useState("");

  // huggingface — token comes from a global secret (default) or a pasted token.
  const [hfSource, setHfSource] = useState<"secret" | "paste">("secret");
  const [hfToken, setHfToken] = useState("");
  const [hfTokenSecret, setHfTokenSecret] = useState("");
  // huggingface — optional custom HF_ENDPOINT (default huggingface.co): none, a
  // pasted URL, or a global secret.
  const [hfEndpointSource, setHfEndpointSource] = useState<"none" | "paste" | "secret">("none");
  const [hfEndpoint, setHfEndpoint] = useState("");
  const [hfEndpointSecret, setHfEndpointSecret] = useState("");
  const [secretKeys, setSecretKeys] = useState<string[]>([]);

  // local
  const [localPath, setLocalPath] = useState("");

  // sftp
  const [sftpHost, setSftpHost] = useState("");
  const [sftpPort, setSftpPort] = useState("22");
  const [sftpUser, setSftpUser] = useState("");
  const [sftpAuth, setSftpAuth] = useState<"password" | "key">("password");
  const [sftpPassword, setSftpPassword] = useState("");
  const [sftpKey, setSftpKey] = useState("");
  const [sftpBasePath, setSftpBasePath] = useState("");

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [test, setTest] = useState<TestState>({ status: "idle" });

  // Global secrets (admin Secrets) the HF token can reference — keys only.
  useEffect(() => {
    let cancel = false;
    fetch("/api/proxy/v1/global-env", { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : []))
      .then((rows: { key: string }[]) => {
        if (!cancel && Array.isArray(rows)) setSecretKeys(rows.map((r) => r.key));
      })
      .catch(() => {
        /* admins only; non-admins just won't see the picker */
      });
    return () => {
      cancel = true;
    };
  }, []);

  // Editing a tested field invalidates a prior pass, so Create re-disables
  // until the user re-tests. No-op when already idle to avoid render churn.
  const invalidateTest = () =>
    setTest((t) => (t.status === "idle" ? t : { status: "idle" }));

  const validate = (): string | null => {
    if (!name.trim()) return "Name is required.";
    if (kind === "s3") {
      if (!bucket.trim()) return "Bucket is required.";
      if (s3CredSource === "secret") {
        const hasOne = accessKeyIdSecret || secretAccessKeySecret;
        const hasBoth = accessKeyIdSecret && secretAccessKeySecret;
        if (hasOne && !hasBoth) {
          return "Pick a secret for both Access key ID and Secret access key, or leave both unset.";
        }
      } else {
        const hasOne = accessKeyId.trim() || secretAccessKey.trim();
        const hasBoth = accessKeyId.trim() && secretAccessKey.trim();
        if (hasOne && !hasBoth) {
          return "Provide both Access key ID and Secret access key, or leave both blank.";
        }
      }
    }
    if (kind === "local" && !localPath.trim()) return "Path is required.";
    if (kind === "sftp") {
      if (!sftpHost.trim()) return "Host is required.";
      if (!sftpUser.trim()) return "Username is required.";
    }
    return null;
  };

  // Build the kind-specific payload shared by Test + Create.
  const kindPayload = () => {
    if (kind === "s3") {
      const creds =
        s3CredSource === "secret"
          ? {
              access_key_id_secret: accessKeyIdSecret || null,
              secret_access_key_secret: secretAccessKeySecret || null,
            }
          : {
              access_key_id: accessKeyId.trim() || null,
              secret_access_key: secretAccessKey.trim() || null,
            };
      return {
        bucket: bucket.trim(),
        prefix: prefix.trim() || null,
        region: region.trim() || null,
        endpoint: endpoint.trim() || null,
        ...creds,
      };
    }
    if (kind === "local") {
      return { path: localPath.trim() };
    }
    if (kind === "sftp") {
      return {
        host: sftpHost.trim(),
        port: Number(sftpPort) || 22,
        username: sftpUser.trim(),
        base_path: sftpBasePath.trim() || null,
        password: sftpAuth === "password" ? sftpPassword || null : null,
        private_key: sftpAuth === "key" ? sftpKey || null : null,
      };
    }
    // huggingface: token (secret/paste) + optional custom endpoint (none/paste/secret).
    const endpointPart =
      hfEndpointSource === "secret"
        ? { endpoint_secret: hfEndpointSecret || null }
        : hfEndpointSource === "paste"
          ? { endpoint: hfEndpoint.trim() || null }
          : {};
    const tokenPart =
      hfSource === "secret"
        ? { hf_token_secret: hfTokenSecret || null }
        : { hf_token: hfToken.trim() || null };
    return { ...tokenPart, ...endpointPart };
  };

  const onTest = async () => {
    setError(null);
    const err = validate();
    if (err) {
      setTest({ status: "fail", message: err });
      return;
    }
    setTest({ status: "running" });
    try {
      const r = await gateway.testStorage({ kind, ...kindPayload() });
      setTest(r.ok ? { status: "ok", message: r.message } : { status: "fail", message: r.message });
    } catch (e) {
      setTest({ status: "fail", message: e instanceof Error ? e.message : String(e) });
    }
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    const err = validate();
    if (err) {
      setError(err);
      return;
    }
    setSubmitting(true);
    try {
      await gateway.createStorage({
        name: name.trim(),
        kind,
        notes: notes.trim() || null,
        ...kindPayload(),
      });
      router.push("/storage");
      router.refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="flex flex-col gap-5">
      <section className="rounded-lg border border-border bg-card p-5">
        <div className="mb-4">
          <h2 className="text-base font-medium">Storage backend</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">
            {kind === "s3"
              ? "An S3 (or S3-compatible: R2, MinIO) bucket the platform writes to."
              : kind === "local"
                ? "A local filesystem path on the gateway host — for hosting catalog repos."
                : kind === "sftp"
                  ? "A remote server reached over SFTP — for hosting catalog repos."
                  : "A HuggingFace token holder for pushing datasets / models to the Hub."}
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <Label htmlFor="storage-kind" className="text-xs uppercase tracking-wide text-muted-foreground">Kind</Label>
            <Select
              value={kind}
              onValueChange={(v) => {
                setKind(v as StorageKind);
                invalidateTest();
              }}
            >
              <SelectTrigger id="storage-kind" className="mt-1.5">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="s3">S3 (or S3-compatible)</SelectItem>
                <SelectItem value="local">Local filesystem</SelectItem>
                <SelectItem value="sftp">SFTP</SelectItem>
                <SelectItem value="huggingface">HuggingFace</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label htmlFor="storage-name" className="text-xs uppercase tracking-wide text-muted-foreground">Name</Label>
            <Input
              id="storage-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={kind === "s3" ? "e.g. prod-s3" : "e.g. hf-scicom"}
              className="mt-1.5"
            />
          </div>
        </div>
      </section>

      {kind === "s3" && (
        <section className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-4 text-base font-medium">Bucket</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div>
              <Label htmlFor="s3-bucket" className="text-xs uppercase tracking-wide text-muted-foreground">Bucket</Label>
              <Input
                id="s3-bucket"
                value={bucket}
                onChange={(e) => {
                  setBucket(e.target.value);
                  invalidateTest();
                }}
                placeholder="gpuplatform"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="s3-prefix" className="text-xs uppercase tracking-wide text-muted-foreground">Prefix (optional)</Label>
              <Input
                id="s3-prefix"
                value={prefix}
                onChange={(e) => setPrefix(e.target.value)}
                placeholder="datasets"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="s3-region" className="text-xs uppercase tracking-wide text-muted-foreground">Region (optional)</Label>
              <Input
                id="s3-region"
                value={region}
                onChange={(e) => {
                  setRegion(e.target.value);
                  invalidateTest();
                }}
                placeholder="ap-southeast-5"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="s3-endpoint" className="text-xs uppercase tracking-wide text-muted-foreground">Endpoint (optional)</Label>
              <Input
                id="s3-endpoint"
                value={endpoint}
                onChange={(e) => {
                  setEndpoint(e.target.value);
                  invalidateTest();
                }}
                placeholder="https://<account>.r2.cloudflarestorage.com"
                className="mt-1.5 font-mono text-xs"
              />
              <p className="mt-1 text-xs text-muted-foreground">
                Set for R2, MinIO, or any S3-compatible API.
              </p>
            </div>
          </div>

          <div className="mt-4 rounded-md border border-border p-3">
            <div className="mb-1 text-sm font-medium">Credentials</div>
            <p className="mb-3 text-xs text-muted-foreground">
              Optional — leave blank to fall back to{" "}
              <span className="font-mono">AWS_ACCESS_KEY_ID</span> /{" "}
              <span className="font-mono">AWS_SECRET_ACCESS_KEY</span> on the gateway.
              Paste keys (stored encrypted with Fernet) or reference{" "}
              <a href="/admin/secrets" className="underline">global secrets</a> (rotate without touching this storage).
            </p>

            <div className="mb-3 inline-flex rounded-md border border-border p-0.5 text-xs">
              {(["paste", "secret"] as const).map((src) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => {
                    setS3CredSource(src);
                    invalidateTest();
                  }}
                  className={
                    "rounded px-2.5 py-1 transition-colors " +
                    (s3CredSource === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {src === "paste" ? "Paste keys" : "Global secrets"}
                </button>
              ))}
            </div>

            {s3CredSource === "secret" ? (
              secretKeys.length > 0 ? (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <div>
                    <Label htmlFor="s3-akid-secret" className="text-xs uppercase tracking-wide text-muted-foreground">Access key ID secret</Label>
                    <Select
                      value={accessKeyIdSecret}
                      onValueChange={(v) => {
                        setAccessKeyIdSecret(v);
                        invalidateTest();
                      }}
                    >
                      <SelectTrigger id="s3-akid-secret" className="mt-1.5">
                        <SelectValue placeholder="Select a secret" />
                      </SelectTrigger>
                      <SelectContent>
                        {secretKeys.map((k) => (
                          <SelectItem key={k} value={k} className="font-mono text-xs">{k}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="s3-sak-secret" className="text-xs uppercase tracking-wide text-muted-foreground">Secret access key secret</Label>
                    <Select
                      value={secretAccessKeySecret}
                      onValueChange={(v) => {
                        setSecretAccessKeySecret(v);
                        invalidateTest();
                      }}
                    >
                      <SelectTrigger id="s3-sak-secret" className="mt-1.5">
                        <SelectValue placeholder="Select a secret" />
                      </SelectTrigger>
                      <SelectContent>
                        {secretKeys.map((k) => (
                          <SelectItem key={k} value={k} className="font-mono text-xs">{k}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-muted-foreground">
                  No global secrets yet. Add them under{" "}
                  <a href="/admin/secrets" className="underline">Secrets</a>, then pick them here — or switch to{" "}
                  <span className="font-medium">Paste keys</span>.
                </p>
              )
            ) : (
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <Label htmlFor="s3-akid" className="text-xs uppercase tracking-wide text-muted-foreground">Access key ID</Label>
                  <Input
                    id="s3-akid"
                    type="password"
                    autoComplete="off"
                    value={accessKeyId}
                    onChange={(e) => {
                      setAccessKeyId(e.target.value);
                      invalidateTest();
                    }}
                    className="mt-1.5 font-mono text-xs"
                  />
                </div>
                <div>
                  <Label htmlFor="s3-sak" className="text-xs uppercase tracking-wide text-muted-foreground">Secret access key</Label>
                  <Input
                    id="s3-sak"
                    type="password"
                    autoComplete="off"
                    value={secretAccessKey}
                    onChange={(e) => {
                      setSecretAccessKey(e.target.value);
                      invalidateTest();
                    }}
                    className="mt-1.5 font-mono text-xs"
                  />
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {kind === "local" && (
        <section className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-4 text-base font-medium">Local path</h2>
          <Label htmlFor="local-path" className="text-xs uppercase tracking-wide text-muted-foreground">Path</Label>
          <Input
            id="local-path"
            value={localPath}
            onChange={(e) => {
              setLocalPath(e.target.value);
              invalidateTest();
            }}
            placeholder="/var/lib/gpuplatform/catalog"
            className="mt-1.5 font-mono text-xs"
          />
          <p className="mt-1.5 text-xs text-muted-foreground">
            An absolute directory on the gateway host. Created if it doesn&apos;t exist; must be writable.
          </p>
        </section>
      )}

      {kind === "sftp" && (
        <section className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-4 text-base font-medium">SFTP server</h2>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="md:col-span-2">
              <Label htmlFor="sftp-host" className="text-xs uppercase tracking-wide text-muted-foreground">Host</Label>
              <Input
                id="sftp-host"
                value={sftpHost}
                onChange={(e) => {
                  setSftpHost(e.target.value);
                  invalidateTest();
                }}
                placeholder="files.example.com"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="sftp-port" className="text-xs uppercase tracking-wide text-muted-foreground">Port</Label>
              <Input
                id="sftp-port"
                value={sftpPort}
                onChange={(e) => {
                  setSftpPort(e.target.value);
                  invalidateTest();
                }}
                placeholder="22"
                className="mt-1.5"
              />
            </div>
            <div>
              <Label htmlFor="sftp-user" className="text-xs uppercase tracking-wide text-muted-foreground">Username</Label>
              <Input
                id="sftp-user"
                value={sftpUser}
                onChange={(e) => {
                  setSftpUser(e.target.value);
                  invalidateTest();
                }}
                placeholder="ubuntu"
                className="mt-1.5"
              />
            </div>
            <div className="md:col-span-2">
              <Label htmlFor="sftp-base" className="text-xs uppercase tracking-wide text-muted-foreground">Base path (optional)</Label>
              <Input
                id="sftp-base"
                value={sftpBasePath}
                onChange={(e) => {
                  setSftpBasePath(e.target.value);
                  invalidateTest();
                }}
                placeholder="/home/ubuntu/catalog"
                className="mt-1.5 font-mono text-xs"
              />
            </div>
          </div>

          <div className="mt-4 rounded-md border border-border p-3">
            <div className="mb-2 text-sm font-medium">Credentials</div>
            <div className="mb-3 inline-flex rounded-md border border-border p-0.5 text-xs">
              {(["password", "key"] as const).map((a) => (
                <button
                  key={a}
                  type="button"
                  onClick={() => {
                    setSftpAuth(a);
                    invalidateTest();
                  }}
                  className={
                    "rounded px-2.5 py-1 transition-colors " +
                    (sftpAuth === a ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {a === "password" ? "Password" : "Private key"}
                </button>
              ))}
            </div>
            {sftpAuth === "password" ? (
              <Input
                type="password"
                autoComplete="off"
                value={sftpPassword}
                onChange={(e) => {
                  setSftpPassword(e.target.value);
                  invalidateTest();
                }}
                placeholder="password"
                className="font-mono text-xs"
              />
            ) : (
              <Textarea
                value={sftpKey}
                onChange={(e) => {
                  setSftpKey(e.target.value);
                  invalidateTest();
                }}
                rows={4}
                placeholder="-----BEGIN OPENSSH PRIVATE KEY-----"
                className="font-mono text-xs"
              />
            )}
            <p className="mt-2 text-xs text-muted-foreground">Stored encrypted at rest with Fernet.</p>
          </div>
        </section>
      )}

      {kind === "huggingface" && (
        <section className="rounded-lg border border-border bg-card p-5">
          <div className="mb-1 text-sm font-medium">HuggingFace API token</div>
          <p className="mb-3 text-xs text-muted-foreground">
            Use a global secret (managed once under{" "}
            <a href="/admin/secrets" className="underline">Secrets</a>, shared, rotate without touching this storage)
            or paste a token (stored encrypted here). Leave blank to fall back to{" "}
            <span className="font-mono">HF_TOKEN</span> on the gateway.
          </p>

          <div className="mb-3 inline-flex rounded-md border border-border p-0.5 text-xs">
            {(["secret", "paste"] as const).map((src) => (
              <button
                key={src}
                type="button"
                onClick={() => {
                  setHfSource(src);
                  invalidateTest();
                }}
                className={
                  "rounded px-2.5 py-1 transition-colors " +
                  (hfSource === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                }
              >
                {src === "secret" ? "Global secret" : "Paste a token"}
              </button>
            ))}
          </div>

          {hfSource === "secret" ? (
            secretKeys.length > 0 ? (
              <>
                <Label htmlFor="hf-secret" className="text-xs uppercase tracking-wide text-muted-foreground">Global secret</Label>
                <Select
                  value={hfTokenSecret}
                  onValueChange={(v) => {
                    setHfTokenSecret(v);
                    invalidateTest();
                  }}
                >
                  <SelectTrigger id="hf-secret" className="mt-1.5">
                    <SelectValue placeholder="Select a secret (e.g. HF_TOKEN)" />
                  </SelectTrigger>
                  <SelectContent>
                    {secretKeys.map((k) => (
                      <SelectItem key={k} value={k} className="font-mono text-xs">
                        {k}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="mt-1.5 text-xs text-muted-foreground">
                  Resolved from Secrets at use-time — rotate it there and this storage picks it up automatically.
                </p>
              </>
            ) : (
              <p className="text-xs text-muted-foreground">
                No global secrets yet. Add one under{" "}
                <a href="/admin/secrets" className="underline">Secrets</a> (e.g. <span className="font-mono">HF_TOKEN</span>),
                then pick it here — or switch to <span className="font-medium">Paste a token</span>.
              </p>
            )
          ) : (
            <>
              <Label htmlFor="hf-token" className="text-xs uppercase tracking-wide text-muted-foreground">Token</Label>
              <Input
                id="hf-token"
                type="password"
                autoComplete="off"
                value={hfToken}
                onChange={(e) => {
                  setHfToken(e.target.value);
                  invalidateTest();
                }}
                placeholder="hf_..."
                className="mt-1.5 font-mono text-xs"
              />
              <p className="mt-1.5 text-xs text-muted-foreground">
                Stored encrypted at rest with Fernet. Generate one at{" "}
                <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="underline">
                  huggingface.co/settings/tokens
                </a>
                .
              </p>
            </>
          )}

          <div className="mt-5 border-t border-border pt-4">
            <div className="mb-1 text-sm font-medium">Custom endpoint (optional)</div>
            <p className="mb-3 text-xs text-muted-foreground">
              Point at a self-hosted, HF-compatible Hub (sets{" "}
              <span className="font-mono">HF_ENDPOINT</span>) instead of{" "}
              <span className="font-mono">huggingface.co</span>. Leave as{" "}
              <span className="font-medium">Default</span> for the public Hub.
            </p>

            <div className="mb-3 inline-flex rounded-md border border-border p-0.5 text-xs">
              {(["none", "paste", "secret"] as const).map((src) => (
                <button
                  key={src}
                  type="button"
                  onClick={() => {
                    setHfEndpointSource(src);
                    invalidateTest();
                  }}
                  className={
                    "rounded px-2.5 py-1 transition-colors " +
                    (hfEndpointSource === src ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {src === "none" ? "Default" : src === "paste" ? "Paste a URL" : "Global secret"}
                </button>
              ))}
            </div>

            {hfEndpointSource === "paste" ? (
              <>
                <Label htmlFor="hf-endpoint" className="text-xs uppercase tracking-wide text-muted-foreground">Endpoint URL</Label>
                <Input
                  id="hf-endpoint"
                  value={hfEndpoint}
                  onChange={(e) => {
                    setHfEndpoint(e.target.value);
                    invalidateTest();
                  }}
                  placeholder="https://hf.internal.example.com"
                  className="mt-1.5 font-mono text-xs"
                />
              </>
            ) : hfEndpointSource === "secret" ? (
              secretKeys.length > 0 ? (
                <>
                  <Label htmlFor="hf-endpoint-secret" className="text-xs uppercase tracking-wide text-muted-foreground">Endpoint secret</Label>
                  <Select
                    value={hfEndpointSecret}
                    onValueChange={(v) => {
                      setHfEndpointSecret(v);
                      invalidateTest();
                    }}
                  >
                    <SelectTrigger id="hf-endpoint-secret" className="mt-1.5">
                      <SelectValue placeholder="Select a secret (e.g. HF_ENDPOINT)" />
                    </SelectTrigger>
                    <SelectContent>
                      {secretKeys.map((k) => (
                        <SelectItem key={k} value={k} className="font-mono text-xs">{k}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="mt-1.5 text-xs text-muted-foreground">
                    Resolved from Secrets at use-time.
                  </p>
                </>
              ) : (
                <p className="text-xs text-muted-foreground">
                  No global secrets yet. Add one under{" "}
                  <a href="/admin/secrets" className="underline">Secrets</a>, then pick it here — or switch to{" "}
                  <span className="font-medium">Paste a URL</span>.
                </p>
              )
            ) : null}
          </div>
        </section>
      )}

      <section className="rounded-lg border border-border bg-card p-5">
        <Label htmlFor="storage-notes" className="text-xs uppercase tracking-wide text-muted-foreground">Notes (optional)</Label>
        <Textarea
          id="storage-notes"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={2}
          className="mt-1.5"
        />
      </section>

      <div className="flex items-center gap-3">
        {error && <span className="text-sm text-destructive">{error}</span>}
        <div className="ml-auto flex items-center gap-3">
          {test.status === "ok" && (
            <span className="text-sm text-emerald-600 dark:text-emerald-400">✓ {test.message}</span>
          )}
          {test.status === "fail" && (
            <span className="text-right text-sm text-destructive">✕ {test.message}</span>
          )}
          <Button
            type="button"
            variant="outline"
            onClick={onTest}
            disabled={test.status === "running" || submitting}
          >
            {test.status === "running" && <Loader2 className="h-4 w-4 animate-spin" />}
            {test.status === "running" ? "Testing…" : "Test"}
          </Button>
          <Button
            type="submit"
            disabled={submitting || test.status !== "ok"}
            title={test.status !== "ok" ? "Pass the connection test first" : undefined}
          >
            {submitting && <Loader2 className="h-4 w-4 animate-spin" />}
            {submitting ? "Creating…" : "Create storage"}
          </Button>
        </div>
      </div>
    </form>
  );
}
