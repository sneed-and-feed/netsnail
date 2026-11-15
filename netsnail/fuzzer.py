"""Cross-ISA Deterministic Differential Chaos Fuzzer (x86_64 SIMD vs Scalar vs WASM32).

Executes identical randomized multi-player trajectories (swept-capsule wall/crease slides,
Cayley leans, crouch/sprint inertia, sub-tick 30-plate ricochets & penetrations, finite-velocity
projectiles, anonymous acoustics, and Minkowski PVS datagram packing) across:
  (a) Native x86_64 / AArch64 Auto-SIMD (`NT_SIMD_MODE_AUTO = 0`),
  (b) Forced Portable Scalar (`NT_SIMD_MODE_FORCE_SCALAR = 1`),
  (c) Compiled `wasm32-unknown-unknown` (`netsnail_rs.wasm` via `wasmtime` or Node.js WebAssembly),
  (d) Hostile x86_64 `MXCSR` Flush-To-Zero (`FTZ`/`DAZ`) caller environment.

Hashes the canonical 1,572,864-byte `NetTwinArena` state at every single tick via both
`nt_hash_arena_deterministic` (64-bit) and cryptographic `BLAKE2b-512` + `BLAKE3-256`,
asserting 0-bit / 0-ULP divergence across >= 100,000 player-tick steps.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from netsnail.arena import (
    NT_ARENA_BYTES,
    NT_BTN_ADS,
    NT_BTN_CROUCH,
    NT_BTN_FIRE,
    NT_BTN_LEAN_L,
    NT_BTN_LEAN_R,
    NT_BTN_SPRINT,
    NT_HIT_FLAG_HIT,
    NT_HIT_FLAG_PENETRATED,
    NT_HIT_FLAG_RICOCHET,
    NT_OFFSET_TELEMETRY_AND_HEADER,
    NT_SIMD_MODE_AUTO,
    NT_SIMD_MODE_FORCE_SCALAR,
    ClientInputCmd,
    NetTwinArenaBridge,
    blake3_256_hex,
    find_rust_library,
    find_wasm_library,
    hostile_mxcsr_flush_to_zero,
)


# ==============================================================================
# 1. Deterministic SplitMix64 Chaos Schedule Generator
# ==============================================================================

def _splitmix64_stream(seed: int, count: int) -> np.ndarray:
    """Generates deterministic uint64 stream using SplitMix64."""
    out = np.empty(count, dtype=np.uint64)
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(count):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
        out[i] = z
    return out


def default_chaos_spawns(num_players: int) -> List[Tuple[int, float, float, float, int]]:
    """Deterministic multi-room + central-pillar spawn ring exercising walls, doorways, and sightlines."""
    spawns: List[Tuple[int, float, float, float, int]] = []
    for p in range(num_players):
        angle = (2.0 * math.pi * p) / max(1, num_players)
        # Alternate inner central arena (around concrete pillar [-1.5,1.5]^2) and outer wing corridors
        radius = 4.5 if (p % 2 == 0) else 13.5
        x = float(round(radius * math.cos(angle), 3))
        y = float(round(radius * math.sin(angle), 3))
        # Ensure nobody spawns inside [-1.5, 1.5]^2 pillar or right inside x=+-10 / y=+-12 wall planes
        if abs(x) < 2.2 and abs(y) < 2.2:
            x = 2.5 if x >= 0 else -2.5
        if 9.2 <= abs(x) <= 10.8:
            x = 8.5 if x >= 0 else -8.5
        if 11.2 <= abs(y) <= 12.8:
            y = 10.5 if y >= 0 else -10.5
        yaw_rad = math.atan2(-y, -x)
        yaw_q15 = int(max(-32768, min(32767, round(yaw_rad / math.pi * 32767.0))))
        spawns.append((p, x, y, 0.0, yaw_q15))
    return spawns


def generate_chaos_command_bytes(
    num_ticks: int,
    num_players: int,
    seed: int = 0xD41C_5E41_2026_0101,
) -> bytes:
    """Generates packed `(num_ticks, num_players)` array of 16-byte `ClientInputCmd` structs.

    Exercises full Q15 trig octants, sprint/crouch inertia, Cayley leans, sub-tick `alpha`
    in `[0, 255]/256`, short-range hitscan, shallow-angle 30-plate ricochets, and long-range
    finite-velocity ballistic projectiles.
    """
    total = num_ticks * num_players
    r0 = _splitmix64_stream(seed, total)
    r1 = _splitmix64_stream(seed ^ 0xA5A5_5A5A_F0F0_0F0F, total)

    buf = bytearray(total * 16)
    idx = 0
    for t in range(num_ticks):
        cmd_tick = t + 1
        for p in range(num_players):
            u = int(r0[idx])
            v = int(r1[idx])

            buttons = 0
            # Periodic stances & leans
            if ((t + p * 3) % 11) < 3:
                buttons |= NT_BTN_SPRINT
            elif ((t + p * 5) % 13) < 3:
                buttons |= NT_BTN_CROUCH
            if ((t + p) % 7) == 0:
                buttons |= NT_BTN_ADS
            lean_sel = (u >> 8) & 0x7
            if lean_sel == 1:
                buttons |= NT_BTN_LEAN_L
            elif lean_sel == 2:
                buttons |= NT_BTN_LEAN_R

            # Fire every 8th player-tick stagger so ricochets, penetrations & projectiles overlap
            if ((t + p * 2) & 7) == 0:
                buttons |= NT_BTN_FIRE

            # Full-range forward/strafe Q15 commands driving swept-capsule wall & crease slides
            move_fwd = int(((u >> 16) & 0xFFFF) - 32768)
            move_str = int(((u >> 32) & 0xFFFF) - 32768)

            # Smooth + chaotic yaw sweep across all octants; bounded pitch
            yaw_q15 = int((((v & 0xFFFF) + t * 517 + p * 4099) & 0xFFFF) - 32768)
            pitch_raw = int((((v >> 16) & 0xFFFF) - 32768) // 4)
            pitch_q15 = max(-14000, min(14000, pitch_raw))

            fire_subtick_u8 = int((v >> 32) & 0xFF)
            weapon_slot_u8 = int(((v >> 40) + p) % 4)

            struct.pack_into(
                "<IHhhhhBB",
                buf,
                idx * 16,
                cmd_tick,
                buttons,
                move_fwd,
                move_str,
                yaw_q15,
                pitch_q15,
                fire_subtick_u8,
                weapon_slot_u8,
            )
            idx += 1

    return bytes(buf)


# ==============================================================================
# 2. Native Execution Runner (SIMD Mode 0 or Forced Scalar Mode 1)
# ==============================================================================

def _run_native_chaos_pass(
    simd_mode: int,
    num_ticks: int,
    num_players: int,
    spawns: List[Tuple[int, float, float, float, int]],
    cmd_bytes: bytes,
    full_arena_hash_every_tick: bool = True,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    arena = NetTwinArenaBridge(auto_init=True)
    assert arena.set_simd_backend_override(simd_mode) == 0
    assert arena.load_default_raid_map() == 0

    for pid, x, y, z, yaw_q15 in spawns:
        assert arena.spawn_player(pid, x, y, z, yaw_q15) == 0
        arena.set_player_rtt_ticks(pid, 6 + (pid % 12))

    total_cmds = num_ticks * num_players
    cmd_arr = (ClientInputCmd * total_cmds).from_buffer_copy(cmd_bytes)

    rolling_u64_blake2b = hashlib.blake2b(digest_size=32)
    rolling_arena_blake2b = hashlib.blake2b(digest_size=32)
    tick_digests_u64 = np.empty(num_ticks, dtype=np.uint64)

    mv = arena.memoryview_bytes()
    off_simd_flag = NT_OFFSET_TELEMETRY_AND_HEADER + 4

    hits_count = 0
    ricochets_count = 0
    penetrations_count = 0
    milestone_digests: Dict[str, Dict[str, str]] = {}
    milestone_set = {
        0,
        max(0, num_ticks // 4 - 1),
        max(0, num_ticks // 2 - 1),
        max(0, (3 * num_ticks) // 4 - 1),
        num_ticks - 1,
    }

    for t in range(num_ticks):
        base = t * num_players
        for p in range(num_players):
            cmd = cmd_arr[base + p]
            rc = arena.ingest_client_cmd(p, cmd)
            if rc < 0:
                raise RuntimeError(f"nt_ingest_client_cmd failed at tick={t}, player={p}: rc={rc}")

        rc_step = arena.step_kinematics()
        if rc_step < 0:
            raise RuntimeError(f"nt_step_kinematics failed at tick={t}: rc={rc_step}")

        for p in range(num_players):
            cmd = cmd_arr[base + p]
            if cmd.buttons_stance & NT_BTN_FIRE:
                hit = arena.resolve_subtick_fire(p, cmd)
                if hit.outcome_flags & NT_HIT_FLAG_HIT:
                    hits_count += 1
                if hit.outcome_flags & NT_HIT_FLAG_RICOCHET:
                    ricochets_count += 1
                if hit.outcome_flags & NT_HIT_FLAG_PENETRATED:
                    penetrations_count += 1

        if (t & 15) == 0:
            obs = t % num_players
            ack_tick = max(0, arena.server_tick - 8)
            _ = arena.pack_client_datagram(obs, ack_tick)

        d64 = arena.hash_arena_deterministic()
        tick_digests_u64[t] = np.uint64(d64)
        rolling_u64_blake2b.update(struct.pack("<Q", d64))

        if full_arena_hash_every_tick:
            h_tick = hashlib.blake2b(digest_size=64)
            h_tick.update(mv[:off_simd_flag])
            h_tick.update(b"\x00\x00\x00\x00")
            h_tick.update(mv[off_simd_flag + 4 :])
            rolling_arena_blake2b.update(h_tick.digest())

        if t in milestone_set:
            milestone_digests[str(t + 1)] = {
                "nt_hash_u64": f"{d64:016x}",
                "arena_blake3_256": arena.canonical_arena_blake3_hex(mask_simd_override=True),
                "arena_blake2b_256": arena.canonical_arena_blake2b_hex(mask_simd_override=True),
            }

    final_blake3 = arena.canonical_arena_blake3_hex(mask_simd_override=True)
    final_blake2b = arena.canonical_arena_blake2b_hex(mask_simd_override=True)
    trajectory_u64_blake3 = blake3_256_hex(tick_digests_u64.view(np.uint8))
    elapsed_s = time.perf_counter() - t0

    hdr = arena.struct.telemetry_and_header
    return {
        "backend": "x86_64_simd" if simd_mode == NT_SIMD_MODE_AUTO else "portable_scalar",
        "simd_mode_override": simd_mode,
        "elapsed_sec": round(elapsed_s, 4),
        "num_ticks": num_ticks,
        "num_players": num_players,
        "player_ticks": num_ticks * num_players,
        "final_nt_hash_u64": f"{int(tick_digests_u64[-1]):016x}",
        "final_arena_blake3_256": final_blake3,
        "final_arena_blake2b_256": final_blake2b,
        "rolling_every_tick_arena_blake2b_256": rolling_arena_blake2b.hexdigest(),
        "rolling_every_tick_u64_blake2b_256": rolling_u64_blake2b.hexdigest(),
        "trajectory_every_tick_u64_blake3_256": trajectory_u64_blake3,
        "tick_digests_u64": tick_digests_u64,
        "milestones": milestone_digests,
        "telemetry": {
            "server_tick": int(hdr.server_tick),
            "hits_count": hits_count,
            "ricochets_count": ricochets_count,
            "penetrations_count": penetrations_count,
            "total_shots_resolved": int(hdr.total_shots_resolved),
            "max_cayley_norm_drift": float(hdr.max_cayley_norm_drift),
        },
    }


# ==============================================================================
# 3. WASM32 Execution Runner (Node.js WebAssembly / wasmtime)
# ==============================================================================

_NODE_WASM_RUNNER_JS = r"""
const fs = require('fs');
const crypto = require('crypto');

async function main() {
  const [wasmPath, configPath, cmdBinPath, outJsonPath] = process.argv.slice(2);
  const cfg = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  const cmdBuf = fs.readFileSync(cmdBinPath);
  const wasmBytes = fs.readFileSync(wasmPath);

  const t0 = performance.now();
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports;
  const mem = ex.memory;

  const arenaPtr = ex.nt_wasm_global_arena() >>> 0;
  const cmdPtr = ex.nt_wasm_cmd_buf() >>> 0;
  const hitPtr = ex.nt_wasm_hit_buf() >>> 0;
  const mtuPtr = ex.nt_wasm_mtu_buf() >>> 0;

  if (ex.nt_arena_init(arenaPtr) !== 0) throw new Error('nt_arena_init failed in WASM');
  if (ex.nt_set_simd_backend_override(arenaPtr, 0) !== 0) throw new Error('nt_set_simd_backend_override failed');
  if (ex.nt_load_default_raid_map(arenaPtr) !== 0) throw new Error('nt_load_default_raid_map failed');

  for (const sp of cfg.spawns) {
    const [pid, x, y, z, yaw] = sp;
    if (ex.nt_spawn_player(arenaPtr, pid, x, y, z, yaw) !== 0) {
      throw new Error(`nt_spawn_player(${pid}) failed in WASM`);
    }
    if (typeof ex.nt_set_player_rtt_ticks === 'function') {
      ex.nt_set_player_rtt_ticks(arenaPtr, pid, 6 + (pid % 12));
    }
  }

  const numTicks = cfg.num_ticks;
  const numPlayers = cfg.num_players;
  const fullEveryTick = Boolean(cfg.full_arena_hash_every_tick);
  const milestoneSet = new Set(cfg.milestone_ticks);

  const offSimdFlag = 1507328 + 4;
  const zero4 = Buffer.alloc(4, 0);

  const rollingU64 = crypto.createHash('blake2b512'); // wait: Python uses digest_size=32 -> use raw 8B buffer!
  const u64RawBuf = Buffer.alloc(numTicks * 8);
  const perTickDigests64 = [];

  let hitsCount = 0;
  let ricochetsCount = 0;
  let penetrationsCount = 0;
  const milestoneArenaSnapshots = {};

  // Pre-allocate 64-byte scratch buffer per tick for rolling arena blake2b512 leaves
  const leafHashesConcat = fullEveryTick ? Buffer.alloc(numTicks * 64) : Buffer.alloc(0);

  for (let t = 0; t < numTicks; t++) {
    const baseOff = t * numPlayers * 16;
    for (let p = 0; p < numPlayers; p++) {
      const cOff = baseOff + p * 16;
      const wasmCmdView = new Uint8Array(mem.buffer, cmdPtr, 16);
      wasmCmdView.set(cmdBuf.subarray(cOff, cOff + 16));
      const rc = ex.nt_ingest_client_cmd(arenaPtr, p, cmdPtr);
      if (rc < 0) throw new Error(`WASM nt_ingest_client_cmd failed at tick=${t} p=${p}: ${rc}`);
    }

    const rcStep = ex.nt_step_kinematics(arenaPtr);
    if (rcStep < 0) throw new Error(`WASM nt_step_kinematics failed at tick=${t}: ${rcStep}`);

    for (let p = 0; p < numPlayers; p++) {
      const cOff = baseOff + p * 16;
      const buttons = cmdBuf.readUInt16LE(cOff + 4);
      if ((buttons & 1) !== 0) {
        const wasmCmdView = new Uint8Array(mem.buffer, cmdPtr, 16);
        wasmCmdView.set(cmdBuf.subarray(cOff, cOff + 16));
        const rcHit = ex.nt_resolve_subtick_fire(arenaPtr, p, cmdPtr, hitPtr);
        if (rcHit < 0) throw new Error(`WASM nt_resolve_subtick_fire failed at tick=${t}: ${rcHit}`);
        const hitFlags = new Uint8Array(mem.buffer, hitPtr + 14, 1)[0];
        if (hitFlags & 1) hitsCount++;
        if (hitFlags & 4) ricochetsCount++;
        if (hitFlags & 2) penetrationsCount++;
      }
    }

    if ((t & 15) === 0) {
      const obs = t % numPlayers;
      const srvTick = new DataView(mem.buffer, arenaPtr + 1507328, 4).getUint32(0, true);
      const ackTick = srvTick > 8 ? (srvTick - 8) : 0;
      const nBytes = ex.nt_pack_client_datagram(arenaPtr, obs, ackTick, mtuPtr, 1200);
      if (nBytes < 0) throw new Error(`WASM nt_pack_client_datagram failed at tick=${t}: ${nBytes}`);
    }

    const rawHash = ex.nt_hash_arena_deterministic(arenaPtr);
    const u64 = BigInt.asUintN(64, rawHash);
    u64RawBuf.writeBigUInt64LE(u64, t * 8);

    if (fullEveryTick) {
      const hTick = crypto.createHash('blake2b512');
      hTick.update(new Uint8Array(mem.buffer, arenaPtr, offSimdFlag));
      hTick.update(zero4);
      hTick.update(new Uint8Array(mem.buffer, arenaPtr + offSimdFlag + 4, 1572864 - (offSimdFlag + 4)));
      hTick.digest().copy(leafHashesConcat, t * 64);
    }

    if (milestoneSet.has(t)) {
      const snapPath = `${outJsonPath}.snap_${t + 1}.bin`;
      const snapCopy = Buffer.from(new Uint8Array(mem.buffer, arenaPtr, 1572864));
      snapCopy.fill(0, offSimdFlag, offSimdFlag + 4);
      fs.writeFileSync(snapPath, snapCopy);
      milestoneArenaSnapshots[String(t + 1)] = {
        nt_hash_u64: u64.toString(16).padStart(16, '0'),
        snap_path: snapPath,
      };
    }
  }

  const finalSnapPath = `${outJsonPath}.final.bin`;
  const finalCopy = Buffer.from(new Uint8Array(mem.buffer, arenaPtr, 1572864));
  finalCopy.fill(0, offSimdFlag, offSimdFlag + 4);
  fs.writeFileSync(finalSnapPath, finalCopy);

  const u64RawPath = `${outJsonPath}.u64.bin`;
  fs.writeFileSync(u64RawPath, u64RawBuf);

  const leafPath = `${outJsonPath}.leaves.bin`;
  if (fullEveryTick) {
    fs.writeFileSync(leafPath, leafHashesConcat);
  }

  const dvHdr = new DataView(mem.buffer, arenaPtr + 1507328, 80);
  const elapsedSec = (performance.now() - t0) / 1000.0;
  const lastU64 = u64RawBuf.readBigUInt64LE((numTicks - 1) * 8);

  fs.writeFileSync(outJsonPath, JSON.stringify({
    elapsed_sec: Number(elapsedSec.toFixed(4)),
    final_nt_hash_u64: lastU64.toString(16).padStart(16, '0'),
    final_snap_path: finalSnapPath,
    u64_raw_path: u64RawPath,
    leaf_hashes_path: fullEveryTick ? leafPath : null,
    milestone_snaps: milestoneArenaSnapshots,
    telemetry: {
      server_tick: dvHdr.getUint32(0, true),
      hits_count: hitsCount,
      ricochets_count: ricochetsCount,
      penetrations_count: penetrationsCount,
      total_shots_resolved: dvHdr.getUint32(28, true),
      max_cayley_norm_drift: dvHdr.getFloat32(64, true),
    }
  }));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""


def _run_wasm32_chaos_pass(
    wasm_path: Path,
    num_ticks: int,
    num_players: int,
    spawns: List[Tuple[int, float, float, float, int]],
    cmd_bytes: bytes,
    full_arena_hash_every_tick: bool = True,
) -> Dict[str, Any]:
    node_exe = shutil.which("node")
    if not node_exe:
        raise RuntimeError("Node.js ('node') is required for WASM32 differential execution")

    milestone_ticks = sorted(
        {
            0,
            max(0, num_ticks // 4 - 1),
            max(0, num_ticks // 2 - 1),
            max(0, (3 * num_ticks) // 4 - 1),
            num_ticks - 1,
        }
    )

    with tempfile.TemporaryDirectory(prefix="netsnail_wasm_fuzz_") as tmp:
        tmpdir = Path(tmp)
        js_path = tmpdir / "runner.js"
        cfg_path = tmpdir / "config.json"
        cmd_path = tmpdir / "cmds.bin"
        out_path = tmpdir / "out.json"

        js_path.write_text(_NODE_WASM_RUNNER_JS, encoding="utf-8")
        cfg_path.write_text(
            json.dumps(
                {
                    "num_ticks": num_ticks,
                    "num_players": num_players,
                    "spawns": spawns,
                    "full_arena_hash_every_tick": full_arena_hash_every_tick,
                    "milestone_ticks": milestone_ticks,
                }
            ),
            encoding="utf-8",
        )
        cmd_path.write_bytes(cmd_bytes)

        proc = subprocess.run(
            [node_exe, str(js_path), str(wasm_path), str(cfg_path), str(cmd_path), str(out_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"WASM32 Node runner failed (rc={proc.returncode}):\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
            )

        meta = json.loads(out_path.read_text(encoding="utf-8"))
        final_arena_bytes = Path(meta["final_snap_path"]).read_bytes()
        u64_raw_bytes = Path(meta["u64_raw_path"]).read_bytes()
        tick_digests_u64 = np.frombuffer(u64_raw_bytes, dtype="<u8").copy()

        final_blake3 = blake3_256_hex(final_arena_bytes)
        final_blake2b = hashlib.blake2b(final_arena_bytes, digest_size=32).hexdigest()
        rolling_u64_blake2b = hashlib.blake2b(u64_raw_bytes, digest_size=32).hexdigest()
        trajectory_u64_blake3 = blake3_256_hex(u64_raw_bytes)

        rolling_arena_blake2b = hashlib.blake2b(digest_size=32)
        if full_arena_hash_every_tick and meta.get("leaf_hashes_path"):
            rolling_arena_blake2b.update(Path(meta["leaf_hashes_path"]).read_bytes())

        milestones: Dict[str, Dict[str, str]] = {}
        for k, v in meta["milestone_snaps"].items():
            sb = Path(v["snap_path"]).read_bytes()
            milestones[k] = {
                "nt_hash_u64": v["nt_hash_u64"],
                "arena_blake3_256": blake3_256_hex(sb),
                "arena_blake2b_256": hashlib.blake2b(sb, digest_size=32).hexdigest(),
            }

        return {
            "backend": "wasm32_simd128",
            "runtime": f"node ({node_exe})",
            "elapsed_sec": meta["elapsed_sec"],
            "num_ticks": num_ticks,
            "num_players": num_players,
            "player_ticks": num_ticks * num_players,
            "final_nt_hash_u64": meta["final_nt_hash_u64"],
            "final_arena_blake3_256": final_blake3,
            "final_arena_blake2b_256": final_blake2b,
            "rolling_every_tick_arena_blake2b_256": rolling_arena_blake2b.hexdigest(),
            "rolling_every_tick_u64_blake2b_256": rolling_u64_blake2b,
            "trajectory_every_tick_u64_blake3_256": trajectory_u64_blake3,
            "tick_digests_u64": tick_digests_u64,
            "milestones": milestones,
            "telemetry": meta["telemetry"],
        }


# ==============================================================================
# 4. MXCSR Flush-To-Zero / Denormals-Are-Zero Resilience Verification
# ==============================================================================

def verify_mxcsr_ftz_resilience(
    num_ticks: int = 512,
    num_players: int = 8,
    seed: int = 0x515D_F720_2026,
) -> Dict[str, Any]:
    """Verifies 0-bit divergence when caller enters C-ABI with hostile MXCSR FTZ enabled."""
    spawns = default_chaos_spawns(num_players)
    cmd_bytes = generate_chaos_command_bytes(num_ticks, num_players, seed=seed)

    clean_res = _run_native_chaos_pass(
        simd_mode=NT_SIMD_MODE_AUTO,
        num_ticks=num_ticks,
        num_players=num_players,
        spawns=spawns,
        cmd_bytes=cmd_bytes,
        full_arena_hash_every_tick=False,
    )

    with hostile_mxcsr_flush_to_zero() as mxcsr_info:
        ftz_res = _run_native_chaos_pass(
            simd_mode=NT_SIMD_MODE_AUTO,
            num_ticks=num_ticks,
            num_players=num_players,
            spawns=spawns,
            cmd_bytes=cmd_bytes,
            full_arena_hash_every_tick=False,
        )

    mismatches = int(np.sum(clean_res["tick_digests_u64"] != ftz_res["tick_digests_u64"]))
    exact_match = (
        mismatches == 0
        and clean_res["final_arena_blake3_256"] == ftz_res["final_arena_blake3_256"]
        and clean_res["trajectory_every_tick_u64_blake3_256"] == ftz_res["trajectory_every_tick_u64_blake3_256"]
    )

    # Also verify caller's FTZ control word was preserved across the FFI calls
    caller_preserved = True
    if mxcsr_info.get("supported"):
        caller_preserved = mxcsr_info.get("during") == mxcsr_info.get("after_ffi_call")

    return {
        "mxcsr_hardware_toggle_supported": bool(mxcsr_info.get("supported")),
        "caller_control_word_preserved": caller_preserved,
        "ticks_tested": num_ticks,
        "players_tested": num_players,
        "player_ticks_tested": num_ticks * num_players,
        "clean_final_blake3_256": clean_res["final_arena_blake3_256"],
        "hostile_ftz_final_blake3_256": ftz_res["final_arena_blake3_256"],
        "tick_digest_mismatches": mismatches,
        "zero_ulp_resilient": exact_match and caller_preserved,
    }


# ==============================================================================
# 5. Top-Level Cross-ISA Differential Chaos Campaign & Receipt Writer
# ==============================================================================

def run_cross_isa_chaos_fuzzer(
    num_ticks: int = 2_500,
    num_players: int = 64,
    seed: int = 0xD41C_5E41_2026_0200,
    full_arena_hash_every_tick: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs x86_64 SIMD vs Scalar vs WASM32 differential chaos campaign and archives receipts."""
    wasm_path = find_wasm_library()
    dll_path = find_rust_library()
    if dll_path is None:
        raise FileNotFoundError("Native netsnail_rs library not found")
    if wasm_path is None:
        raise FileNotFoundError("WASM32 netsnail_rs.wasm module not found")

    spawns = default_chaos_spawns(num_players)
    cmd_bytes = generate_chaos_command_bytes(num_ticks, num_players, seed=seed)

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_simd = pool.submit(
            _run_native_chaos_pass,
            NT_SIMD_MODE_AUTO,
            num_ticks,
            num_players,
            spawns,
            cmd_bytes,
            full_arena_hash_every_tick,
        )
        fut_scalar = pool.submit(
            _run_native_chaos_pass,
            NT_SIMD_MODE_FORCE_SCALAR,
            num_ticks,
            num_players,
            spawns,
            cmd_bytes,
            full_arena_hash_every_tick,
        )
        fut_wasm = pool.submit(
            _run_wasm32_chaos_pass,
            wasm_path,
            num_ticks,
            num_players,
            spawns,
            cmd_bytes,
            full_arena_hash_every_tick,
        )
        res_simd = fut_simd.result()
        res_scalar = fut_scalar.result()
        res_wasm = fut_wasm.result()

    mxcsr_report = verify_mxcsr_ftz_resilience()
    wall_sec = time.perf_counter() - t_start

    arr_simd = res_simd.pop("tick_digests_u64")
    arr_scalar = res_scalar.pop("tick_digests_u64")
    arr_wasm = res_wasm.pop("tick_digests_u64")

    diff_simd_scalar = np.nonzero(arr_simd != arr_scalar)[0]
    diff_simd_wasm = np.nonzero(arr_simd != arr_wasm)[0]

    mismatched_ticks_simd_vs_scalar = int(diff_simd_scalar.size)
    mismatched_ticks_simd_vs_wasm = int(diff_simd_wasm.size)
    first_divergence_tick: Optional[int] = None
    if diff_simd_scalar.size > 0:
        first_divergence_tick = int(diff_simd_scalar[0]) + 1
    elif diff_simd_wasm.size > 0:
        first_divergence_tick = int(diff_simd_wasm[0]) + 1

    blake3_match = (
        res_simd["final_arena_blake3_256"]
        == res_scalar["final_arena_blake3_256"]
        == res_wasm["final_arena_blake3_256"]
    )
    rolling_arena_match = (
        res_simd["rolling_every_tick_arena_blake2b_256"]
        == res_scalar["rolling_every_tick_arena_blake2b_256"]
        == res_wasm["rolling_every_tick_arena_blake2b_256"]
    )
    trajectory_blake3_match = (
        res_simd["trajectory_every_tick_u64_blake3_256"]
        == res_scalar["trajectory_every_tick_u64_blake3_256"]
        == res_wasm["trajectory_every_tick_u64_blake3_256"]
    )

    zero_ulp_verified = (
        mismatched_ticks_simd_vs_scalar == 0
        and mismatched_ticks_simd_vs_wasm == 0
        and blake3_match
        and rolling_arena_match
        and trajectory_blake3_match
        and mxcsr_report["zero_ulp_resilient"]
    )

    receipt: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "verifier": "NetSnail Cross-ISA BLAKE3 Differential Chaos Fuzzer (netsnail/fuzzer.py)",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed_hex": f"0x{seed:016x}",
        "arena_size_bytes": NT_ARENA_BYTES,
        "num_players": num_players,
        "num_ticks": num_ticks,
        "total_player_ticks": num_players * num_ticks,
        "full_arena_hashed_every_tick": full_arena_hash_every_tick,
        "wall_time_sec": round(wall_sec, 4),
        "artifacts": {
            "native_library": str(dll_path).replace("\\", "/"),
            "wasm32_module": str(wasm_path).replace("\\", "/"),
        },
        "backends": {
            "x86_64_simd": res_simd,
            "portable_scalar": res_scalar,
            "wasm32": res_wasm,
        },
        "parity_summary": {
            "digest_x86_simd_blake3_256": res_simd["final_arena_blake3_256"],
            "digest_scalar_blake3_256": res_scalar["final_arena_blake3_256"],
            "digest_wasm32_blake3_256": res_wasm["final_arena_blake3_256"],
            "rolling_every_tick_arena_blake2b_256": res_simd["rolling_every_tick_arena_blake2b_256"],
            "trajectory_every_tick_u64_blake3_256": res_simd["trajectory_every_tick_u64_blake3_256"],
            "mismatched_ticks_simd_vs_scalar": mismatched_ticks_simd_vs_scalar,
            "mismatched_ticks_simd_vs_wasm32": mismatched_ticks_simd_vs_wasm,
            "first_divergence_tick": first_divergence_tick,
            "bit_divergence_count": mismatched_ticks_simd_vs_scalar + mismatched_ticks_simd_vs_wasm,
            "ulp_divergence": 0 if zero_ulp_verified else -1,
            "mxcsr_ftz_resilience": mxcsr_report,
            "zero_ulp_cross_isa_verified": zero_ulp_verified,
            "status": "PASS" if zero_ulp_verified else "FAIL",
        },
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    if not zero_ulp_verified:
        raise AssertionError(
            f"Cross-ISA 0-ULP parity FAILED! first_divergence_tick={first_divergence_tick}, "
            f"simd_vs_scalar={mismatched_ticks_simd_vs_scalar}, simd_vs_wasm={mismatched_ticks_simd_vs_wasm}, "
            f"blake3_match={blake3_match}, rolling_match={rolling_arena_match}, mxcsr={mxcsr_report}"
        )

    return receipt
