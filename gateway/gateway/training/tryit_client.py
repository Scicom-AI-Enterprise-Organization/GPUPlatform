#!/usr/bin/env python3
"""Tiny client for the persistent Try-it server (tryit_server.py). The gateway
SSH-execs this to forward one request to the loaded model over its Unix socket and
print the JSON response on stdout (one line). Keeps no ML deps — plain stdlib."""
import argparse
import socket
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--req", required=True, help="path to the JSON request file")
    a = ap.parse_args()
    with open(a.req) as f:
        payload = f.read().strip().encode() + b"\n"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(600)
    s.connect(a.sock)
    s.sendall(payload)
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    s.close()
    sys.stdout.write(buf.decode())
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
