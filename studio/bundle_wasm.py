#!/usr/bin/env python3
"""Bundle compiled netsnail_rs.wasm into studio/netsnail_portable_studio.html.

Encodes target/wasm32-unknown-unknown/release/netsnail_rs.wasm as an inline
base64 constant (NETSNAIL_WASM_B64) so the 4-pane WebGL2 Tactical Netcode
Oscilloscope runs 100% offline over file:// with zero external requests.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import re
import sys


STUDIO_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = STUDIO_DIR.parent
DEFAULT_HTML = STUDIO_DIR / "netsnail_portable_studio.html"
DEFAULT_WASM_CANDIDATES = [
    REPO_ROOT / "target" / "wasm32-unknown-unknown" / "release" / "netsnail_rs.wasm",
    REPO_ROOT / "rust" / "netsnail_rs" / "target" / "wasm32-unknown-unknown" / "release" / "netsnail_rs.wasm",
]


def _read_leb128_u32(buf: bytes, offset: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[offset]
        offset += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, offset
        shift += 7


def inspect_wasm_binary(wasm_bytes: bytes) -> dict[str, object]:
    if len(wasm_bytes) < 8 or wasm_bytes[:4] != b"\x00asm":
        raise ValueError("Invalid WebAssembly binary magic header")
    version = int.from_bytes(wasm_bytes[4:8], "little")
    if version != 1:
        raise ValueError(f"Unsupported WebAssembly version: {version}")

    imports: list[str] = []
    exports: list[str] = []
    offset = 8
    n = len(wasm_bytes)
    while offset < n:
        sec_id = wasm_bytes[offset]
        offset += 1
        sec_len, offset = _read_leb128_u32(wasm_bytes, offset)
        sec_end = offset + sec_len
        if sec_id == 2:  # Import section
            count, p = _read_leb128_u32(wasm_bytes, offset)
            for _ in range(count):
                mod_len, p = _read_leb128_u32(wasm_bytes, p)
                mod_name = wasm_bytes[p : p + mod_len].decode("utf-8", errors="replace")
                p += mod_len
                fld_len, p = _read_leb128_u32(wasm_bytes, p)
                fld_name = wasm_bytes[p : p + fld_len].decode("utf-8", errors="replace")
                p += fld_len
                kind = wasm_bytes[p]
                p += 1
                if kind == 0:
                    _, p = _read_leb128_u32(wasm_bytes, p)
                elif kind == 1:
                    p += 1
                    flags, p = _read_leb128_u32(wasm_bytes, p)
                    _, p = _read_leb128_u32(wasm_bytes, p)
                    if flags & 0x1:
                        _, p = _read_leb128_u32(wasm_bytes, p)
                elif kind == 2:
                    flags, p = _read_leb128_u32(wasm_bytes, p)
                    _, p = _read_leb128_u32(wasm_bytes, p)
                    if flags & 0x1:
                        _, p = _read_leb128_u32(wasm_bytes, p)
                elif kind == 3:
                    p += 2
                imports.append(f"{mod_name}.{fld_name}")
        elif sec_id == 7:  # Export section
            count, p = _read_leb128_u32(wasm_bytes, offset)
            for _ in range(count):
                name_len, p = _read_leb128_u32(wasm_bytes, p)
                name = wasm_bytes[p : p + name_len].decode("utf-8", errors="replace")
                p += name_len
                p += 1  # export kind
                _, p = _read_leb128_u32(wasm_bytes, p)  # export index
                exports.append(name)
        offset = sec_end

    forbidden = [imp for imp in imports if any(k in imp for k in ("malloc", "calloc", "realloc", "free", "__rust_alloc"))]
    if forbidden:
        raise RuntimeError(f"WASM binary contains forbidden allocator imports: {forbidden}")

    return {
        "wasm_size_bytes": len(wasm_bytes),
        "sha256": hashlib.sha256(wasm_bytes).hexdigest(),
        "imports": imports,
        "exports": exports,
    }


def bundle_wasm(wasm_path: pathlib.Path, html_path: pathlib.Path) -> dict[str, object]:
    wasm_bytes = wasm_path.read_bytes()
    meta = inspect_wasm_binary(wasm_bytes)
    b64_str = base64.b64encode(wasm_bytes).decode("ascii")

    html_text = html_path.read_text(encoding="utf-8")
    pattern = re.compile(r'const\s+NETSNAIL_WASM_B64\s*=\s*"[^"]*";')
    if not pattern.search(html_text):
        raise RuntimeError(f"Could not find `const NETSNAIL_WASM_B64 = \"...\";` in {html_path}")

    updated = pattern.sub(f'const NETSNAIL_WASM_B64 = "{b64_str}";', html_text, count=1)

    meta_pattern = re.compile(r"const\s+NETSNAIL_WASM_BUILD_META\s*=\s*\{[^\n]*\};")
    meta_payload = json.dumps(
        {
            "wasm_bytes": meta["wasm_size_bytes"],
            "sha256_prefix": str(meta["sha256"])[:16],
            "export_count": len(meta["exports"]),
            "import_count": len(meta["imports"]),
        },
        separators=(",", ":"),
    )
    if meta_pattern.search(updated):
        updated = meta_pattern.sub(f"const NETSNAIL_WASM_BUILD_META = {meta_payload};", updated, count=1)

    # Audit zero external network URLs (http:// or https:// in HTML/CSS/JS)
    ext_urls = re.findall(r"https?://[^\s\"'`<>)]+", updated)
    if ext_urls:
        raise RuntimeError(f"Found external URLs in offline studio HTML: {ext_urls}")

    html_path.write_text(updated, encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Bundle netsnail_rs.wasm into netsnail_portable_studio.html")
    parser.add_argument("--wasm", type=pathlib.Path, default=None, help="Path to compiled netsnail_rs.wasm")
    parser.add_argument("--html", type=pathlib.Path, default=DEFAULT_HTML, help="Path to studio HTML")
    args = parser.parse_args()

    wasm_path = args.wasm
    if wasm_path is None:
        for cand in DEFAULT_WASM_CANDIDATES:
            if cand.is_file():
                wasm_path = cand
                break
    if wasm_path is None or not wasm_path.is_file():
        print("ERROR: netsnail_rs.wasm not found yet. Build with: cargo build --target wasm32-unknown-unknown --release", file=sys.stderr)
        return 1

    meta = bundle_wasm(wasm_path, args.html)
    print(
        f"Bundled {wasm_path.name} ({meta['wasm_size_bytes']} B, sha256={str(meta['sha256'])[:16]}..., "
        f"{len(meta['exports'])} exports, {len(meta['imports'])} imports) -> {args.html}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
