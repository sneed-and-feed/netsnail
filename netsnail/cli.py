"""Command-line interface for NetSnail evaluation, chaos fuzzing, and hardware audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from netsnail.fuzzer import run_cross_isa_chaos_fuzzer
from netsnail.network_sim import (
    run_scenario_a_honest_wan_and_ge_wifi,
    run_scenario_b_clumsy_lag_switch_red_team,
    run_scenario_c_behind_pillar_occluder_cutoff,
    run_scenario_d_anti_dma_and_doorway_pop_in,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="netsnail",
        description="NetSnail v0.1 Deterministic 128 Hz Causal Twin & Anti-DMA Red-Team CLI",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_eval = sub.add_parser("evaluate", help="Run Scenario A-D head-to-head red-team evaluation")
    p_eval.add_argument("--ticks", type=int, default=2048, help="Scenario A simulation ticks")

    p_fuzz = sub.add_parser("fuzz", help="Run Cross-ISA x86_64 SIMD vs Scalar vs WASM32 chaos fuzzer")
    p_fuzz.add_argument("--ticks", type=int, default=10_000, help="Number of ticks")
    p_fuzz.add_argument("--players", type=int, default=16, help="Number of active players")
    p_fuzz.add_argument(
        "--output",
        type=str,
        default="results/cross_isa_blake3_receipts.json",
        help="Receipt output path",
    )

    args = parser.parse_args()
    if args.cmd == "fuzz":
        res = run_cross_isa_chaos_fuzzer(
            num_ticks=args.ticks,
            num_players=args.players,
            output_path=args.output,
        )
        print(json.dumps(res["parity_summary"], indent=2))
        return 0

    # Default or `evaluate`: run Scenario A-D quick summary
    ticks = getattr(args, "ticks", 1024)
    summary = {
        "scenario_a": run_scenario_a_honest_wan_and_ge_wifi(num_ticks=ticks),
        "scenario_b": run_scenario_b_clumsy_lag_switch_red_team(num_trials=25),
        "scenario_c": run_scenario_c_behind_pillar_occluder_cutoff(num_trials=25),
        "scenario_d": run_scenario_d_anti_dma_and_doorway_pop_in(num_dma_ticks=64),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
