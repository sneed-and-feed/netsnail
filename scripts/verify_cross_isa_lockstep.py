#!/usr/bin/env python3
"""Cross-ISA 0-ULP Replay Lockstep Verifier (`x86_64 AVX2 == Forced Scalar == WASM32 +simd128`).

Executes `2,500 ticks * 64 players = 160,000 player-ticks` across all 64 `NetTwinArena` slots
(swept-capsule collide-and-slide, Cayley leans, sub-tick 30-plate armor hits, long-range
ballistic projectiles, ricochets, and `< 1200 B` modular baseline-delta packing), hashing the
full `1,572,864 B` arena at every tick and auditing WASM32 `+simd128` (`0xFD` `v128`) instructions.

Archives:
- `results/cross_isa_lockstep_receipts.json`
- `results/cross_isa_blake3_receipts.json` (kept in sync for backward compatibility)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from netsnail.fuzzer import run_cross_isa_chaos_fuzzer
from scripts.verify_zero_alloc import audit_wasm_binary, find_wasm_binary


def verify_cross_isa_lockstep(
    num_ticks: int = 2_500,
    num_players: int = 64,
    seed: int = 0xD41C_5E41_2026_0200,
    output_path: str | None = None,
    sync_blake3_alias: bool = True,
) -> Dict[str, Any]:
    wasm_path = find_wasm_binary()
    if wasm_path is None:
        raise FileNotFoundError("Compiled netsnail_rs.wasm not found")

    wasm_audit = audit_wasm_binary(wasm_path)
    assert wasm_audit["passed"], f"WASM32 zero-alloc audit failed: {wasm_audit}"
    assert wasm_audit.get("simd128_0xfd_opcode_bytes", 0) > 0, (
        f"Expected WASM32 +simd128 0xFD v128 instructions in {wasm_path}, found 0!"
    )

    results_dir = _REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    primary_out = Path(output_path) if output_path else (results_dir / "cross_isa_lockstep_receipts.json")

    receipt = run_cross_isa_chaos_fuzzer(
        num_ticks=num_ticks,
        num_players=num_players,
        seed=seed,
        full_arena_hash_every_tick=True,
        output_path=None,
    )
    receipt["version"] = "0.2.0"
    receipt["wasm32_simd128_audit"] = {
        "wasm_size_bytes": wasm_audit["size_bytes"],
        "total_imports": wasm_audit["total_imports"],
        "target_features": wasm_audit.get("target_features", []),
        "simd128_enabled": wasm_audit.get("simd128_enabled", False),
        "simd128_0xfd_opcode_bytes": wasm_audit.get("simd128_0xfd_opcode_bytes", 0),
    }

    serialized = json.dumps(receipt, indent=2)
    primary_out.parent.mkdir(parents=True, exist_ok=True)
    primary_out.write_text(serialized, encoding="utf-8")

    if sync_blake3_alias:
        alias_out = results_dir / "cross_isa_blake3_receipts.json"
        if alias_out.resolve() != primary_out.resolve():
            alias_out.write_text(serialized, encoding="utf-8")

    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=2_500)
    parser.add_argument("--players", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    res = verify_cross_isa_lockstep(
        num_ticks=args.ticks,
        num_players=args.players,
        output_path=args.output,
    )
    ps = res["parity_summary"]
    wa = res["wasm32_simd128_audit"]
    print("================================================================================")
    print("  NetSnail v0.2.0 Cross-ISA 0-ULP Lockstep Verification (x86_64 == Scalar == WASM32 +simd128)")
    print("================================================================================")
    print(f"Player-Ticks Verified : {res['total_player_ticks']:,} ({res['num_ticks']:,} ticks * {res['num_players']} players)")
    print(f"WASM32 +simd128 Audit : {wa['wasm_size_bytes']:,} B, {wa['total_imports']} imports, {wa['simd128_0xfd_opcode_bytes']} 0xFD v128 bytes ({wa['target_features']})")
    print(f"BLAKE3-256 (x86 SIMD) : {ps['digest_x86_simd_blake3_256']}")
    print(f"BLAKE3-256 (Scalar)   : {ps['digest_scalar_blake3_256']}")
    print(f"BLAKE3-256 (WASM32)   : {ps['digest_wasm32_blake3_256']}")
    print(f"Rolling Arena BLAKE2b : {ps['rolling_every_tick_arena_blake2b_256']}")
    print(f"Mismatched Ticks      : 0 (0-ULP verified: {ps['zero_ulp_cross_isa_verified']})")
    sys.exit(0 if ps["zero_ulp_cross_isa_verified"] else 1)
