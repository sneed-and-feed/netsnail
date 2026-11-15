#!/usr/bin/env python3
"""Full End-to-End Red-Team Evaluation & Cross-ISA Chaos Campaign Runner for NetSnail v0.1.

Executes:
1. Scenario A: Honest 30 ms fiber vs 160 ms cross-continent + 20 ms Gilbert-Elliott bursty Wi-Fi,
   measuring 32-tick `nt_client_reconcile` latency (< 90 us) and UDP datagram byte sizes (< 1200 B MTU).
2. Scenario B: Deliberate 200 ms `clumsy` lag-switch packet-clumping corner-swing red-team.
3. Scenario C: Anchor ducking behind solid concrete pillar [-1.5, 1.5]^2 vs 180 ms late Peeker.
4. Scenario D: $150 PCIe DMA physical memory reader adversary across 15 occluded enemies behind walls
   (0 out-of-cone leak bits, 0 doorway pop-in frames, 100% acoustic entity-ID stripping).
5. Cross-ISA BLAKE3 Differential Chaos Fuzzer (16 players * 10,000 ticks = 160,000 player-ticks across
   x86_64 SIMD, Portable Scalar, and WASM32 + MXCSR FTZ resilience).
6. Broadcom BCM2712 Cortex-A76 50,000,000-reference cache & zero-alloc static binary verification.

Archives:
- `results/netsnail_evaluation_summary.json`
- `results/cross_isa_blake3_receipts.json`
- `results/bcm2712_cache_receipts.json`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from netsnail.arena import NT_ARENA_ALIGN, NT_ARENA_BYTES, NT_MTU_CEILING_BYTES, NT_REWIND_RING_BYTES
from netsnail.fuzzer import run_cross_isa_chaos_fuzzer
from netsnail.network_sim import (
    run_scenario_a_honest_wan_and_ge_wifi,
    run_scenario_b_clumsy_lag_switch_red_team,
    run_scenario_c_behind_pillar_occluder_cutoff,
    run_scenario_d_anti_dma_and_doorway_pop_in,
)
from scripts.verify_bcm2712_arena import verify_bcm2712_arena
from scripts.verify_zero_alloc import audit_zero_alloc


def evaluate_all(
    chaos_ticks: int = 10_000,
    chaos_players: int = 16,
    cache_refs: int = 50_000_000,
) -> Dict[str, Any]:
    results_dir = _REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("================================================================================")
    print("  NetSnail v0.1 Head-to-Head Red-Team & Cross-ISA Verification Suite")
    print("================================================================================")

    print("\n[1/6] Auditing #![no_std] Zero-Allocation Native & WASM32 Binaries...")
    zero_alloc = audit_zero_alloc(require_both=True)
    assert zero_alloc["overall_passed"], f"Zero-alloc binary audit failed: {zero_alloc}"

    print("\n[2/6] Running Scenario A (30ms Fiber vs 160ms+20ms Gilbert-Elliott Wi-Fi & 32-Tick Reconcile)...")
    scen_a = run_scenario_a_honest_wan_and_ge_wifi(num_ticks=2048)
    print(
        f"  -> 32-tick nt_client_reconcile: mean={scen_a['reconcile_32_tick_us']['mean_us']:.2f} us, "
        f"p50={scen_a['reconcile_32_tick_us']['median_p50_us']:.2f} us, "
        f"p99={scen_a['reconcile_32_tick_us']['p99_us']:.2f} us (target < 90 us)"
    )
    print(
        f"  -> UDP Datagram Bytes: mean={scen_a['udp_datagram_bytes']['netsnail_mean_bytes']} B, "
        f"max={scen_a['udp_datagram_bytes']['netsnail_max_bytes']} B (< {NT_MTU_CEILING_BYTES} B MTU) "
        f"vs Naive={scen_a['udp_datagram_bytes']['naive_30hz_full_scene_bytes']} B"
    )

    print("\n[3/6] Running Scenario B (200ms Clumsy Lag-Switch Packet-Clumping Red-Team)...")
    scen_b = run_scenario_b_clumsy_lag_switch_red_team(num_trials=50)
    print(
        f"  -> Lag-Switch Exploit Rejection: NetSnail={scen_b['netsnail_lag_switch_exploit_rejection_rate_pct']:.1f}% "
        f"(quarantine={scen_b['netsnail_quarantine_detection_rate_pct']:.1f}%) "
        f"vs Naive={scen_b['naive_lag_switch_exploit_rejection_rate_pct']:.1f}%"
    )

    print("\n[4/6] Running Scenario C (Anchor Behind Concrete Pillar vs 180ms Late Peeker)...")
    scen_c = run_scenario_c_behind_pillar_occluder_cutoff(num_trials=50)
    print(
        f"  -> Behind-Wall Death Rate: NetSnail={scen_c['netsnail_behind_wall_death_rate_pct']:.2f}% "
        f"({scen_c['netsnail_occluder_cutoff_rejections']}/{scen_c['trials']} occluder cutoffs) "
        f"vs Naive={scen_c['naive_behind_wall_death_rate_pct']:.1f}%"
    )

    print("\n[5/6] Running Scenario D ($150 PCIe DMA Adversary + 15 Occluded Enemies + Doorway Pop-In)...")
    scen_d = run_scenario_d_anti_dma_and_doorway_pop_in(num_dma_ticks=128)
    print(
        f"  -> Out-of-Cone DMA Leak Bits: NetSnail wire={scen_d['netsnail_out_of_cone_dma_wire_leak_bits']} bits, "
        f"RAM={scen_d['netsnail_out_of_cone_dma_ram_leak_bits']} bits "
        f"vs Naive={scen_d['naive_out_of_cone_dma_wire_leak_bits']:,} bits"
    )
    print(
        f"  -> Doorway Pop-In Frames: NetSnail={scen_d['doorway_swing']['netsnail_pop_in_frames']} frames "
        f"| Acoustic Entity-ID Stripping: {scen_d['acoustic_entity_id_stripping_pct']:.1f}%"
    )

    print(
        f"\n[6/6] Running Cross-ISA BLAKE3 Chaos Fuzzer ({chaos_players} players * {chaos_ticks:,} ticks = "
        f"{chaos_players * chaos_ticks:,} player-ticks) & BCM2712 50M-Ref Cache Verifier..."
    )
    chaos_receipt_path = str(results_dir / "cross_isa_blake3_receipts.json")
    chaos_res = run_cross_isa_chaos_fuzzer(
        num_ticks=chaos_ticks,
        num_players=chaos_players,
        full_arena_hash_every_tick=True,
        output_path=chaos_receipt_path,
    )
    bcm_receipt_path = str(results_dir / "bcm2712_cache_receipts.json")
    bcm_res = verify_bcm2712_arena(num_references=cache_refs, output_path=bcm_receipt_path)

    total_wall_sec = time.perf_counter() - t0

    head_to_head = {
        "Naive30Hz_250msRewind_FullScene": {
            "tick_hz": 30,
            "base_pipeline_latency_ms": scen_a["pipeline_latency_ms"]["naive_30hz_tick_plus_100ms_lerp_ms"],
            "fiber_30ms_e2e_latency_ms": scen_a["pipeline_latency_ms"]["fiber_30ms_e2e_naive_ms"],
            "datagram_bytes_16p": scen_a["udp_datagram_bytes"]["naive_30hz_full_scene_bytes"],
            "behind_wall_death_rate_pct": scen_c["naive_behind_wall_death_rate_pct"],
            "lag_switch_exploit_rejection_rate_pct": scen_b["naive_lag_switch_exploit_rejection_rate_pct"],
            "out_of_cone_dma_leak_bits": scen_d["naive_out_of_cone_dma_wire_leak_bits"],
            "out_of_cone_dma_leak_pct": scen_d["naive_out_of_cone_leak_pct"],
            "doorway_pop_in_frames": scen_d["doorway_swing"]["naive_visibility_only_30hz_pop_in_frames"],
            "acoustic_entity_id_stripping_pct": 0.0,
        },
        "NetSnail128Hz_CausalTwin": {
            "tick_hz": 128,
            "base_pipeline_latency_ms": scen_a["pipeline_latency_ms"]["netsnail_128hz_tick_plus_2tick_lerp_ms"],
            "fiber_30ms_e2e_latency_ms": scen_a["pipeline_latency_ms"]["fiber_30ms_e2e_netsnail_ms"],
            "reconcile_32_tick_mean_us": scen_a["reconcile_32_tick_us"]["mean_us"],
            "reconcile_32_tick_p50_us": scen_a["reconcile_32_tick_us"]["median_p50_us"],
            "reconcile_32_tick_p99_us": scen_a["reconcile_32_tick_us"]["p99_us"],
            "datagram_mean_bytes_16p": scen_a["udp_datagram_bytes"]["netsnail_mean_bytes"],
            "datagram_max_bytes_16p": scen_a["udp_datagram_bytes"]["netsnail_max_bytes"],
            "behind_wall_death_rate_pct": scen_c["netsnail_behind_wall_death_rate_pct"],
            "lag_switch_exploit_rejection_rate_pct": scen_b["netsnail_lag_switch_exploit_rejection_rate_pct"],
            "lag_switch_quarantine_detection_rate_pct": scen_b["netsnail_quarantine_detection_rate_pct"],
            "out_of_cone_dma_leak_bits": (
                scen_d["netsnail_out_of_cone_dma_wire_leak_bits"]
                + scen_d["netsnail_out_of_cone_dma_ram_leak_bits"]
            ),
            "out_of_cone_dma_leak_pct": scen_d["netsnail_out_of_cone_leak_pct"],
            "doorway_pop_in_frames": scen_d["doorway_swing"]["netsnail_pop_in_frames"],
            "acoustic_entity_id_stripping_pct": scen_d["acoustic_entity_id_stripping_pct"],
        },
    }

    gates_passed = (
        scen_a["reconcile_32_tick_us"]["within_90us_budget"]
        and scen_a["udp_datagram_bytes"]["strictly_below_1200b_mtu"]
        and scen_c["honest_exposed_control_hit_rate_pct"] == 100.0
        and scen_c["netsnail_behind_wall_death_rate_pct"] == 0.0
        and scen_c["netsnail_occluder_cutoff_rejections"] == scen_c["trials"]
        and scen_c["naive_behind_wall_death_rate_pct"] >= 95.0
        and scen_b["netsnail_lag_switch_exploit_rejection_rate_pct"] >= 95.0
        and scen_d["netsnail_out_of_cone_dma_wire_leak_bits"] == 0
        and scen_d["netsnail_out_of_cone_dma_ram_leak_bits"] == 0
        and scen_d["doorway_swing"]["netsnail_pop_in_frames"] == 0
        and scen_d["acoustic_entity_id_stripping_pct"] == 100.0
        and chaos_res["parity_summary"]["zero_ulp_cross_isa_verified"]
        and bcm_res["summary"]["overall_verification_status"] == "PASS"
        and zero_alloc["overall_passed"]
    )

    summary_doc: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "spoke": "netsnail",
        "version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluation_wall_time_sec": round(total_wall_sec, 3),
        "arena_invariants": {
            "arena_size_bytes": NT_ARENA_BYTES,
            "arena_alignment_bytes": NT_ARENA_ALIGN,
            "rewind_ring_bytes": NT_REWIND_RING_BYTES,
            "mtu_ceiling_bytes": NT_MTU_CEILING_BYTES,
            "dynamic_heap_allocations": 0,
        },
        "head_to_head_comparison": head_to_head,
        "scenarios": {
            "scenario_a_honest_wan_and_ge_wifi": scen_a,
            "scenario_b_clumsy_200ms_lag_switch": scen_b,
            "scenario_c_behind_pillar_occluder_cutoff": scen_c,
            "scenario_d_anti_dma_and_doorway_pvs": scen_d,
        },
        "cross_isa_chaos_summary": chaos_res["parity_summary"],
        "bcm2712_cache_summary": bcm_res["summary"],
        "zero_alloc_binary_audit": {
            "native_passed": zero_alloc["native"]["passed"],
            "wasm32_passed": zero_alloc["wasm32"]["passed"],
            "native_size_bytes": zero_alloc["native"]["size_bytes"],
            "wasm32_size_bytes": zero_alloc["wasm32"]["size_bytes"],
            "heap_allocators_imported": 0,
        },
        "all_acceptance_gates_passed": gates_passed,
        "status": "PASS" if gates_passed else "FAIL",
    }

    summary_path = results_dir / "netsnail_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")
    print(f"\n[Saved] Full evaluation summary archived to: {summary_path}")
    assert gates_passed, f"One or more acceptance gates failed in evaluation summary: {summary_doc['status']}"
    return summary_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chaos-ticks", type=int, default=10_000)
    parser.add_argument("--chaos-players", type=int, default=16)
    parser.add_argument("--cache-refs", type=int, default=50_000_000)
    args = parser.parse_args()

    doc = evaluate_all(
        chaos_ticks=args.chaos_ticks,
        chaos_players=args.chaos_players,
        cache_refs=args.cache_refs,
    )
    sys.exit(0 if doc["all_acceptance_gates_passed"] else 1)
