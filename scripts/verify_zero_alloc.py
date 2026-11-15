#!/usr/bin/env python3
"""Zero-Allocation Static Binary Auditor for NetSnail (#![no_std] Native PE/ELF/Mach-O + WASM32).

Inspects:
1. Native shared library (`netsnail_rs.dll` / `libnetsnail_rs.so` / `libnetsnail_rs.dylib`)
   PE32+ import/export directory tables or ELF64 dynamic symbol tables.
2. WebAssembly module (`netsnail_rs.wasm`) binary Section 2 (Import Section) and Section 7 (Export Section).

Asserts:
- 0 dynamic heap allocator imports (`malloc`, `calloc`, `realloc`, `free`, `aligned_alloc`,
  `posix_memalign`, `_aligned_malloc`, `_aligned_free`, `HeapAlloc`, `HeapReAlloc`, `HeapFree`,
  `RtlAllocateHeap`, `__rust_alloc`, `__rust_dealloc`, `__rust_realloc`, `__rust_alloc_zeroed`).
- Presence of canonical NetSnail `nt_*` C-ABI exports and `nt_wasm_*` static bridge exports.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import pefile  # type: ignore
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False


FORBIDDEN_ALLOCATORS = frozenset(
    {
        "malloc",
        "calloc",
        "realloc",
        "free",
        "aligned_alloc",
        "posix_memalign",
        "_aligned_malloc",
        "_aligned_free",
        "heapalloc",
        "heaprealloc",
        "heapfree",
        "rtlallocateheap",
        "rtlreallocateheap",
        "rtlfreeheap",
        "__rust_alloc",
        "__rust_dealloc",
        "__rust_realloc",
        "__rust_alloc_zeroed",
    }
)

REQUIRED_NATIVE_EXPORTS: Tuple[str, ...] = (
    "nt_arena_init",
    "nt_set_simd_backend_override",
    "nt_load_default_raid_map",
    "nt_spawn_player",
    "nt_get_player_pose",
    "nt_ingest_client_cmd",
    "nt_step_kinematics",
    "nt_client_reconcile",
    "nt_resolve_subtick_fire",
    "nt_pack_client_datagram",
    "nt_unpack_client_datagram",
    "nt_hash_arena_deterministic",
    "nt_sin_q15",
    "nt_cos_q15",
    "nt_atan2_f32",
    "nt_sqrt_f32",
    "nt_dot16_pairwise_f32",
    "nt_cayley_rotor_so3_f32",
)

REQUIRED_WASM_BRIDGE_EXPORTS: Tuple[str, ...] = (
    "nt_wasm_global_arena",
    "nt_wasm_cmd_buf",
    "nt_wasm_mtu_buf",
    "nt_wasm_hit_buf",
    "nt_wasm_acoustic_buf",
)


def find_native_library(repo_root: Optional[Path] = None) -> Optional[Path]:
    root = repo_root or Path(__file__).resolve().parent.parent
    system = platform.system()
    if system == "Windows":
        names = ["netsnail_rs.dll"]
    elif system == "Darwin":
        names = ["libnetsnail_rs.dylib", "netsnail_rs.dylib"]
    else:
        names = ["libnetsnail_rs.so", "netsnail_rs.so"]

    # Also allow cross-platform inspection if another artifact is present
    for extra in ("netsnail_rs.dll", "libnetsnail_rs.so", "libnetsnail_rs.dylib"):
        if extra not in names:
            names.append(extra)

    for lib_name in names:
        candidates = [
            root / "target" / "release" / lib_name,
            root / "rust" / "netsnail_rs" / "target" / "release" / lib_name,
            root / "netsnail" / lib_name,
        ]
        for c in candidates:
            if c.is_file():
                return c.resolve()
    return None


def find_wasm_binary(repo_root: Optional[Path] = None) -> Optional[Path]:
    root = repo_root or Path(__file__).resolve().parent.parent
    candidates = [
        root / "target" / "wasm32-unknown-unknown" / "release" / "netsnail_rs.wasm",
        root / "rust" / "netsnail_rs" / "target" / "wasm32-unknown-unknown" / "release" / "netsnail_rs.wasm",
        root / "netsnail" / "netsnail_rs.wasm",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def _normalize_symbol(sym: str) -> List[str]:
    base = sym.strip().split("@", 1)[0]
    if base.startswith("__imp_"):
        base = base[len("__imp_") :]
    forms = [base.lower()]
    if base.startswith("_") and not base.startswith("__rust_") and not base.startswith("_aligned_"):
        forms.append(base[1:].lower())
    return forms


def _match_forbidden_allocator(sym: str) -> Optional[str]:
    for cand in _normalize_symbol(sym):
        if cand in FORBIDDEN_ALLOCATORS:
            return cand
        if "_zn5alloc5alloc" in cand:
            return cand
    return None


def parse_pe_tables(content: bytes) -> Optional[Tuple[List[str], List[str], List[str]]]:
    """Pure-Python PE32/PE32+ import DLL, imported function, and exported function parser."""
    if len(content) < 64 or content[:2] != b"MZ":
        return None
    try:
        e_lfanew = struct.unpack_from("<I", content, 0x3C)[0]
        if e_lfanew + 24 > len(content) or content[e_lfanew : e_lfanew + 4] != b"PE\x00\x00":
            return None
        _, num_sections, _, _, _, size_opt_hdr, _ = struct.unpack_from("<HHIIIHH", content, e_lfanew + 4)
        opt_off = e_lfanew + 24
        if opt_off + size_opt_hdr > len(content) or size_opt_hdr < 96:
            return None
        magic = struct.unpack_from("<H", content, opt_off)[0]
        if magic == 0x20B:
            dd_off = opt_off + 112
            is_64 = True
        elif magic == 0x10B:
            dd_off = opt_off + 96
            is_64 = False
        else:
            return None
        if dd_off + 16 > opt_off + size_opt_hdr:
            return None

        export_rva, _ = struct.unpack_from("<II", content, dd_off)
        import_rva, _ = struct.unpack_from("<II", content, dd_off + 8)

        sec_base = opt_off + size_opt_hdr
        sections = []
        for i in range(num_sections):
            s_off = sec_base + i * 40
            if s_off + 40 > len(content):
                return None
            vsize, va, raw_size, raw_ptr = struct.unpack_from("<IIII", content, s_off + 8)
            sections.append((va, max(vsize, raw_size), raw_ptr))

        def rva_to_off(rva: int) -> Optional[int]:
            if rva == 0:
                return None
            for va, span, raw_ptr in sections:
                if va <= rva < va + span:
                    off = raw_ptr + (rva - va)
                    return off if 0 <= off < len(content) else None
            return rva if 0 <= rva < len(content) else None

        def read_cstr(off: Optional[int]) -> Optional[str]:
            if off is None or off < 0 or off >= len(content):
                return None
            end = content.find(b"\x00", off)
            if end == -1:
                return None
            return content[off:end].decode("utf-8", errors="ignore")

        imported_dlls: List[str] = []
        imported_syms: List[str] = []
        exported_syms: List[str] = []

        if export_rva != 0:
            exp_off = rva_to_off(export_rva)
            if exp_off is not None and exp_off + 40 <= len(content):
                num_names, _, names_rva, _ = struct.unpack_from("<IIII", content, exp_off + 24)
                names_off = rva_to_off(names_rva)
                if names_off is not None:
                    for i in range(num_names):
                        if names_off + (i + 1) * 4 > len(content):
                            break
                        name_rva = struct.unpack_from("<I", content, names_off + i * 4)[0]
                        sym = read_cstr(rva_to_off(name_rva))
                        if sym:
                            exported_syms.append(sym)

        if import_rva != 0:
            imp_off = rva_to_off(import_rva)
            if imp_off is not None:
                while imp_off + 20 <= len(content):
                    orig_thunk, _, _, name_rva, first_thunk = struct.unpack_from("<IIIII", content, imp_off)
                    if orig_thunk == 0 and name_rva == 0 and first_thunk == 0:
                        break
                    dll_name = read_cstr(rva_to_off(name_rva))
                    if dll_name:
                        imported_dlls.append(dll_name)
                    thunk_rva = orig_thunk if orig_thunk != 0 else first_thunk
                    thunk_off = rva_to_off(thunk_rva)
                    step = 8 if is_64 else 4
                    ord_flag = (1 << 63) if is_64 else (1 << 31)
                    fmt = "<Q" if is_64 else "<I"
                    if thunk_off is not None:
                        while thunk_off + step <= len(content):
                            val = struct.unpack_from(fmt, content, thunk_off)[0]
                            if val == 0:
                                break
                            if (val & ord_flag) == 0:
                                hn_off = rva_to_off(val & 0x7FFFFFFF)
                                if hn_off is not None and hn_off + 2 < len(content):
                                    sym = read_cstr(hn_off + 2)
                                    if sym:
                                        imported_syms.append(sym)
                            thunk_off += step
                    imp_off += 20

        return imported_dlls, imported_syms, exported_syms
    except Exception:
        return None


def parse_elf_tables(content: bytes) -> Optional[Tuple[List[str], List[str]]]:
    """Pure-Python ELF32/ELF64 symbol parser returning (undefined_imports, defined_exports)."""
    if len(content) < 52 or content[:4] != b"\x7fELF" or content[5] != 1:
        return None
    elf_class = content[4]
    try:
        undef_syms: List[str] = []
        defined_syms: List[str] = []
        found = False

        if elf_class == 2:
            if len(content) < 64:
                return None
            e_shoff = struct.unpack_from("<Q", content, 40)[0]
            e_shentsize, e_shnum, _ = struct.unpack_from("<HHH", content, 58)
            if e_shoff == 0 or e_shentsize < 64:
                return None
            if e_shnum == 0 and e_shoff + 64 <= len(content):
                e_shnum = struct.unpack_from("<Q", content, e_shoff + 32)[0]
            shdrs = []
            for i in range(e_shnum):
                off = e_shoff + i * e_shentsize
                if off + 64 > len(content):
                    return None
                shdrs.append(struct.unpack_from("<IIQQQQIIQQ", content, off))
            for sh in shdrs:
                _, sh_type, _, _, sh_offset, sh_size, sh_link, _, _, sh_entsize = sh
                if sh_type in (2, 11) and 0 <= sh_link < len(shdrs):
                    str_sh = shdrs[sh_link]
                    str_off, str_sz = str_sh[4], str_sh[5]
                    if str_off + str_sz > len(content) or sh_offset + sh_size > len(content):
                        continue
                    found = True
                    strtab = content[str_off : str_off + str_sz]
                    step = sh_entsize if sh_entsize >= 24 else 24
                    for off in range(sh_offset, sh_offset + sh_size - step + 1, step):
                        st_name, _, _, st_shndx, _, _ = struct.unpack_from("<IBBHQQ", content, off)
                        if 0 < st_name < len(strtab):
                            end = strtab.find(b"\x00", st_name)
                            if end != -1:
                                raw_name = strtab[st_name:end].decode("utf-8", errors="ignore")
                                base_name = raw_name.split("@", 1)[0]
                                if base_name:
                                    if st_shndx == 0:
                                        undef_syms.append(base_name)
                                    else:
                                        defined_syms.append(base_name)
        elif elf_class == 1:
            e_shoff = struct.unpack_from("<I", content, 32)[0]
            e_shentsize, e_shnum, _ = struct.unpack_from("<HHH", content, 46)
            if e_shoff == 0 or e_shentsize < 40:
                return None
            shdrs32 = []
            for i in range(e_shnum):
                off = e_shoff + i * e_shentsize
                if off + 40 > len(content):
                    return None
                shdrs32.append(struct.unpack_from("<IIIIIIIIII", content, off))
            for sh in shdrs32:
                _, sh_type, _, _, sh_offset, sh_size, sh_link, _, _, sh_entsize = sh
                if sh_type in (2, 11) and 0 <= sh_link < len(shdrs32):
                    str_sh = shdrs32[sh_link]
                    str_off, str_sz = str_sh[4], str_sh[5]
                    if str_off + str_sz > len(content) or sh_offset + sh_size > len(content):
                        continue
                    found = True
                    strtab = content[str_off : str_off + str_sz]
                    step = sh_entsize if sh_entsize >= 16 else 16
                    for off in range(sh_offset, sh_offset + sh_size - step + 1, step):
                        st_name, _, _, _, _, st_shndx = struct.unpack_from("<IIIBBH", content, off)
                        if 0 < st_name < len(strtab):
                            end = strtab.find(b"\x00", st_name)
                            if end != -1:
                                raw_name = strtab[st_name:end].decode("utf-8", errors="ignore")
                                base_name = raw_name.split("@", 1)[0]
                                if base_name:
                                    if st_shndx == 0:
                                        undef_syms.append(base_name)
                                    else:
                                        defined_syms.append(base_name)
        if not found:
            return None
        return undef_syms, defined_syms
    except Exception:
        return None


def _read_uleb128(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        if pos >= len(buf):
            raise ValueError("Truncated LEB128 in WASM binary")
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7
        if shift > 35:
            raise ValueError("LEB128 overflow in WASM binary")


def parse_wasm_module(content: bytes) -> Dict[str, Any]:
    """Parses WebAssembly binary magic/version, Section 2 (Imports), and Section 7 (Exports)."""
    if len(content) < 8 or content[:4] != b"\x00asm" or content[4:8] != b"\x01\x00\x00\x00":
        raise ValueError("Invalid WebAssembly magic header or version")

    pos = 8
    imports: List[Dict[str, str]] = []
    exports: List[Dict[str, Any]] = []
    sections_seen: List[int] = []

    kind_names = {0x00: "func", 0x01: "table", 0x02: "memory", 0x03: "global", 0x04: "tag"}

    while pos < len(content):
        sec_id = content[pos]
        pos += 1
        sec_len, pos = _read_uleb128(content, pos)
        sec_end = pos + sec_len
        if sec_end > len(content):
            raise ValueError(f"Truncated WASM section {sec_id}")
        sections_seen.append(sec_id)

        if sec_id == 2:  # Import Section
            cur = pos
            count, cur = _read_uleb128(content, cur)
            for _ in range(count):
                mod_len, cur = _read_uleb128(content, cur)
                mod_name = content[cur : cur + mod_len].decode("utf-8", errors="replace")
                cur += mod_len
                fld_len, cur = _read_uleb128(content, cur)
                fld_name = content[cur : cur + fld_len].decode("utf-8", errors="replace")
                cur += fld_len
                kind = content[cur]
                cur += 1
                if kind == 0x00:  # func
                    _, cur = _read_uleb128(content, cur)
                elif kind == 0x01:  # table
                    cur += 1  # elem reftype
                    flags, cur = _read_uleb128(content, cur)
                    _, cur = _read_uleb128(content, cur)
                    if flags & 0x01:
                        _, cur = _read_uleb128(content, cur)
                elif kind == 0x02:  # memory
                    flags, cur = _read_uleb128(content, cur)
                    _, cur = _read_uleb128(content, cur)
                    if flags & 0x01:
                        _, cur = _read_uleb128(content, cur)
                elif kind == 0x03:  # global
                    cur += 2  # valtype + mutability
                elif kind == 0x04:  # tag
                    cur += 1
                    _, cur = _read_uleb128(content, cur)
                imports.append(
                    {
                        "module": mod_name,
                        "name": fld_name,
                        "kind": kind_names.get(kind, f"unknown({kind})"),
                    }
                )
        elif sec_id == 7:  # Export Section
            cur = pos
            count, cur = _read_uleb128(content, cur)
            for _ in range(count):
                nm_len, cur = _read_uleb128(content, cur)
                nm = content[cur : cur + nm_len].decode("utf-8", errors="replace")
                cur += nm_len
                kind = content[cur]
                cur += 1
                idx, cur = _read_uleb128(content, cur)
                exports.append(
                    {
                        "name": nm,
                        "kind": kind_names.get(kind, f"unknown({kind})"),
                        "index": idx,
                    }
                )

        pos = sec_end

    return {
        "size_bytes": len(content),
        "sections_seen": sections_seen,
        "imports": imports,
        "exports": exports,
    }


def audit_native_binary(dll_path: Path) -> Dict[str, Any]:
    content = dll_path.read_bytes()
    imported_dlls: List[str] = []
    imported_syms: List[str] = []
    exported_syms: List[str] = []
    binary_format = "UNKNOWN"

    if content[:2] == b"MZ":
        binary_format = "PE32+"
        if PEFILE_AVAILABLE:
            try:
                pe = pefile.PE(data=content, fast_load=False)
                if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        if entry.dll:
                            imported_dlls.append(entry.dll.decode("utf-8", errors="ignore"))
                        for imp in entry.imports:
                            if imp.name:
                                imported_syms.append(imp.name.decode("utf-8", errors="ignore"))
                if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
                    for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                        if exp.name:
                            exported_syms.append(exp.name.decode("utf-8", errors="ignore"))
            except Exception:
                pass
        if not exported_syms and not imported_syms:
            parsed = parse_pe_tables(content)
            if parsed is not None:
                imported_dlls, imported_syms, exported_syms = parsed
    elif content[:4] == b"\x7fELF":
        binary_format = "ELF"
        parsed_elf = parse_elf_tables(content)
        if parsed_elf is not None:
            imported_syms, exported_syms = parsed_elf
            # Also scan defined symbols for internal __rust_alloc definitions
            for dsym in exported_syms:
                if dsym.startswith("__rust_alloc") or dsym.startswith("__rust_dealloc") or dsym.startswith("__rust_realloc"):
                    imported_syms.append(dsym)
    else:
        binary_format = "MACH-O"
        proc = subprocess.run(["nm", "-u", str(dll_path)], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                parts = line.strip().split()
                if parts:
                    imported_syms.append(parts[-1])
        proc_exp = subprocess.run(["nm", "-gU", str(dll_path)], capture_output=True, text=True, check=False)
        if proc_exp.returncode == 0:
            for line in proc_exp.stdout.splitlines():
                parts = line.strip().split()
                if parts:
                    exported_syms.append(parts[-1].lstrip("_"))

    forbidden_hits: List[str] = []
    for dll_nm in imported_dlls:
        if "crt-heap" in dll_nm.lower():
            forbidden_hits.append(f"dll:{dll_nm}")
    for sym in imported_syms:
        hit = _match_forbidden_allocator(sym)
        if hit is not None:
            forbidden_hits.append(sym)

    forbidden_hits = sorted(set(forbidden_hits))
    export_set = set(exported_syms)
    missing_exports = [s for s in REQUIRED_NATIVE_EXPORTS if s not in export_set]

    return {
        "audited": True,
        "path": str(dll_path).replace("\\", "/"),
        "format": binary_format,
        "size_bytes": len(content),
        "imported_dlls": sorted(set(imported_dlls)),
        "total_imported_symbols": len(imported_syms),
        "total_exported_symbols": len(exported_syms),
        "heap_allocators_imported": forbidden_hits,
        "missing_required_exports": missing_exports,
        "passed": len(forbidden_hits) == 0 and len(missing_exports) == 0,
    }


def audit_wasm_binary(wasm_path: Path) -> Dict[str, Any]:
    content = wasm_path.read_bytes()
    parsed = parse_wasm_module(content)
    imports = parsed["imports"]
    exports = parsed["exports"]

    forbidden_hits: List[str] = []
    for imp in imports:
        hit = _match_forbidden_allocator(imp["name"])
        if hit is not None:
            forbidden_hits.append(f"{imp['module']}.{imp['name']}")

    export_names = {e["name"] for e in exports if e["kind"] == "func"}
    missing_core = [s for s in REQUIRED_NATIVE_EXPORTS if s not in export_names]
    missing_bridge = [s for s in REQUIRED_WASM_BRIDGE_EXPORTS if s not in export_names]

    return {
        "audited": True,
        "path": str(wasm_path).replace("\\", "/"),
        "format": "WASM32",
        "size_bytes": parsed["size_bytes"],
        "imports": imports,
        "total_imports": len(imports),
        "total_exports": len(exports),
        "heap_allocators_imported": sorted(set(forbidden_hits)),
        "missing_core_exports": missing_core,
        "missing_wasm_bridge_exports": missing_bridge,
        "passed": len(forbidden_hits) == 0 and len(missing_core) == 0 and len(missing_bridge) == 0,
    }


def audit_zero_alloc(
    dll_path: Optional[Path] = None,
    wasm_path: Optional[Path] = None,
    require_both: bool = True,
) -> Dict[str, Any]:
    resolved_dll = dll_path or find_native_library()
    resolved_wasm = wasm_path or find_wasm_binary()

    native_res: Dict[str, Any] = (
        audit_native_binary(resolved_dll)
        if resolved_dll and resolved_dll.is_file()
        else {"audited": False, "passed": False, "error": "Native shared library not found"}
    )
    wasm_res: Dict[str, Any] = (
        audit_wasm_binary(resolved_wasm)
        if resolved_wasm and resolved_wasm.is_file()
        else {"audited": False, "passed": False, "error": "WASM32 module not found"}
    )

    if require_both:
        overall_passed = bool(native_res.get("passed")) and bool(wasm_res.get("passed"))
    else:
        overall_passed = bool(native_res.get("passed") or wasm_res.get("passed"))

    return {
        "native": native_res,
        "wasm32": wasm_res,
        "overall_passed": overall_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=str, default=None, help="Explicit path to native shared library")
    parser.add_argument("--wasm", type=str, default=None, help="Explicit path to netsnail_rs.wasm")
    args = parser.parse_args()

    res = audit_zero_alloc(
        dll_path=Path(args.dll) if args.dll else None,
        wasm_path=Path(args.wasm) if args.wasm else None,
        require_both=True,
    )

    print("================================================================================")
    print("  NetSnail #![no_std] Zero-Allocation Static Binary Audit (Native + WASM32)")
    print("================================================================================")

    nat = res["native"]
    if nat.get("audited"):
        print(f"[Native] Path              : {nat['path']} ({nat['format']}, {nat['size_bytes']:,} B)")
        print(f"[Native] Imported DLLs     : {nat['imported_dlls']}")
        print(f"[Native] Heap Allocators   : {nat['heap_allocators_imported']} (count={len(nat['heap_allocators_imported'])})")
        print(f"[Native] Exported Symbols  : {nat['total_exported_symbols']} (missing={nat['missing_required_exports']})")
        print(f"[Native] Gate              : {'PASS' if nat['passed'] else 'FAIL'}")
    else:
        print(f"[Native] Gate              : FAIL ({nat.get('error')})")

    print("--------------------------------------------------------------------------------")
    wsm = res["wasm32"]
    if wsm.get("audited"):
        print(f"[WASM32] Path              : {wsm['path']} ({wsm['size_bytes']:,} B)")
        print(f"[WASM32] Module Imports    : {wsm['imports']} (count={wsm['total_imports']})")
        print(f"[WASM32] Heap Allocators   : {wsm['heap_allocators_imported']} (count={len(wsm['heap_allocators_imported'])})")
        print(
            f"[WASM32] Module Exports    : {wsm['total_exports']} "
            f"(missing_core={wsm['missing_core_exports']}, missing_bridge={wsm['missing_wasm_bridge_exports']})"
        )
        print(f"[WASM32] Gate              : {'PASS' if wsm['passed'] else 'FAIL'}")
    else:
        print(f"[WASM32] Gate              : FAIL ({wsm.get('error')})")

    print("================================================================================")
    if res["overall_passed"]:
        print("PASS: 0 dynamic heap allocator imports across both Native and WASM32 binaries!")
        return 0
    print("FAIL: Zero-allocation binary audit failed!", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
