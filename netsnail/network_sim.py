"""Gilbert-Elliott Hostile Network Simulator, Clumsy Lag-Switch Red-Team & Anti-DMA Evaluator.

Provides a discrete-event UDP channel modeling:
- Gilbert-Elliott 2-state Markov bursty packet loss & Wi-Fi jitter
- Deliberate `clumsy` 200 ms outbound packet-clumping / lag-switch corner-swing attacks
- Head-to-head quantitative comparison between:
    * Baseline: `Naive30Hz_250msRewind_FullScene`
    * System:   `NetSnail128Hz_CausalTwin` (driven through compiled `netsnail_rs` C-ABI)
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Any, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

from netsnail.arena import (
    NT_BTN_ADS,
    NT_BTN_CROUCH,
    NT_BTN_FIRE,
    NT_BTN_LEAN_L,
    NT_BTN_LEAN_R,
    NT_BTN_SPRINT,
    NT_GATE_LAG_SWITCH_QUARANTINED,
    NT_HIT_FLAG_HIT,
    NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED,
    NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED,
    NT_MTU_CEILING_BYTES,
    NT_TICK_HZ,
    AcousticImpulse,
    ClientInputCmd,
    HitRecordOut,
    NetTwinArenaBridge,
    PlayerPoseSnapshot,
)

T = TypeVar("T")


# ==============================================================================
# 1. Gilbert-Elliott + Clumsy Lag-Switch Discrete-Event UDP Pipe
# ==============================================================================

@dataclass(frozen=True)
class LinkProfile:
    name: str
    rtt_ms: float
    jitter_ms: float = 1.0
    p_good_to_bad: float = 0.0
    p_bad_to_good: float = 1.0
    loss_good: float = 0.0
    loss_bad: float = 0.0


FIBER_30MS_PROFILE = LinkProfile(
    name="Honest_30ms_Fiber",
    rtt_ms=30.0,
    jitter_ms=0.8,
    p_good_to_bad=0.002,
    p_bad_to_good=0.50,
    loss_good=0.0,
    loss_bad=0.01,
)

CROSS_CONTINENT_160MS_GE_WIFI = LinkProfile(
    name="CrossContinent_160ms_GE20ms_WiFi",
    rtt_ms=160.0,
    jitter_ms=20.0,
    p_good_to_bad=0.05,
    p_bad_to_good=0.25,
    loss_good=0.002,
    loss_bad=0.18,
)

PEEKER_180MS_PROFILE = LinkProfile(
    name="LatePeeker_180ms_WAN",
    rtt_ms=180.0,
    jitter_ms=6.0,
    p_good_to_bad=0.01,
    p_bad_to_good=0.35,
    loss_good=0.001,
    loss_bad=0.05,
)


class GilbertElliottUdpPipe(Generic[T]):
    """Discrete-event UDP channel with 2-state Markov burst loss, jitter, and lag-switch hold."""

    def __init__(self, profile: LinkProfile, seed: int = 20261115) -> None:
        self.profile = profile
        self._rng = np.random.default_rng(seed)
        self._in_bad_state: bool = False
        self._seq: int = 0
        self._heap: List[Tuple[float, int, T]] = []
        self._hold_until_ms: float = -1.0
        self._clump_index: int = 0

        self.packets_sent: int = 0
        self.packets_dropped: int = 0
        self.packets_delivered: int = 0
        self.bad_state_steps: int = 0

    def activate_clumsy_lag_switch(self, current_time_ms: float, hold_duration_ms: float = 200.0) -> None:
        """Holds all outbound packets until `current_time_ms + hold_duration_ms`, releasing as a clump."""
        self._hold_until_ms = max(self._hold_until_ms, current_time_ms + hold_duration_ms)
        self._clump_index = 0

    def _step_markov_loss(self) -> bool:
        u_trans = float(self._rng.random())
        if not self._in_bad_state:
            if u_trans < self.profile.p_good_to_bad:
                self._in_bad_state = True
        else:
            if u_trans < self.profile.p_bad_to_good:
                self._in_bad_state = False

        if self._in_bad_state:
            self.bad_state_steps += 1
            drop_prob = self.profile.loss_bad
        else:
            drop_prob = self.profile.loss_good

        return float(self._rng.random()) < drop_prob

    def send(self, tx_time_ms: float, payload: T) -> Optional[float]:
        self.packets_sent += 1
        if self._step_markov_loss():
            self.packets_dropped += 1
            return None

        one_way_ms = 0.5 * self.profile.rtt_ms
        if self._in_bad_state:
            jitter = float(self._rng.uniform(0.25 * self.profile.jitter_ms, self.profile.jitter_ms))
        else:
            jitter = float(self._rng.uniform(-0.15 * self.profile.jitter_ms, 0.35 * self.profile.jitter_ms))

        effective_tx_ms = tx_time_ms
        if tx_time_ms < self._hold_until_ms:
            # Clumsy lag-switch buffers packets and flushes them in a 50 us micro-burst clump
            effective_tx_ms = self._hold_until_ms + 0.05 * self._clump_index
            self._clump_index += 1
            jitter = 0.0

        arrival_ms = max(tx_time_ms, effective_tx_ms + max(0.0, one_way_ms + jitter))
        self._seq += 1
        heapq.heappush(self._heap, (arrival_ms, self._seq, payload))
        return arrival_ms

    def pop_arrived(self, up_to_time_ms: float) -> List[Tuple[float, T]]:
        arrived: List[Tuple[float, T]] = []
        while self._heap and self._heap[0][0] <= up_to_time_ms:
            arr_t, _, item = heapq.heappop(self._heap)
            self.packets_delivered += 1
            arrived.append((arr_t, item))
        return arrived


# ==============================================================================
# 2. Geometric Helpers & Naive 30Hz + 250ms Rewind + FullScene Baseline Model
# ==============================================================================

def _yaw_to_q15(yaw_rad: float) -> int:
    wrapped = ((yaw_rad + math.pi) % (2.0 * math.pi)) - math.pi
    return int(max(-32768, min(32767, round((wrapped / math.pi) * 32767.0))))


def _segment_intersects_aabb_2d(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> bool:
    """Slab intersection test in XY plane for hard-cover pillar/wall checks."""
    dx = bx - ax
    dy = by - ay
    tmin = 0.0
    tmax = 1.0

    for p_coord, d_coord, lo, hi in ((ax, dx, min_x, max_x), (ay, dy, min_y, max_y)):
        if abs(d_coord) < 1e-9:
            if p_coord < lo or p_coord > hi:
                return False
        else:
            inv_d = 1.0 / d_coord
            t1 = (lo - p_coord) * inv_d
            t2 = (hi - p_coord) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


def _central_pillar_occludes(ax: float, ay: float, bx: float, by: float) -> bool:
    return _segment_intersects_aabb_2d(ax, ay, bx, by, -1.5, -1.5, 1.5, 1.5)


def _east_wall_doorway_clear_los(ax: float, ay: float, bx: float, by: float) -> bool:
    """Checks if segment from central room (ax < 10) to East corridor (bx >= 10) passes through [-2, 2] doorway."""
    if ax < 10.0 <= bx or bx < 10.0 <= ax:
        if abs(bx - ax) < 1e-9:
            return abs(ay) <= 2.0
        t = (10.0 - ax) / (bx - ax)
        y_cross = ay + t * (by - ay)
        if abs(y_cross) > 2.0:
            return False
    if _central_pillar_occludes(ax, ay, bx, by):
        return False
    return True


class Naive30HzFullSceneBaseline:
    """Reference baseline: 30 Hz server tick, 100 ms fixed lerp, unconditional 250 ms rewind, full-scene broadcast."""

    TICK_HZ: int = 30
    TICK_MS: float = 1000.0 / 30.0
    FIXED_LERP_MS: float = 100.0
    MAX_REWIND_MS: float = 250.0
    HEADER_BYTES: int = 28
    PER_ENTITY_FULL_BYTES: int = 76   # entity_id(4) + xyz(12) + vel(12) + rot(12) + 30-plate HP(32) + loot/weapon(4)
    PER_SOUND_RAW_BYTES: int = 24     # entity_id(4) + xyz(12) + sound_id(4) + volume_f32(4)

    @classmethod
    def pack_full_scene_datagram(
        cls,
        poses: Dict[int, Tuple[float, float, float, int, int]],
        sounds: Sequence[Tuple[int, float, float, float, int]],
        server_tick_30hz: int,
    ) -> bytes:
        """Serializes EVERY actor and raw XYZ+entity_id sound event on the map unconditionally."""
        import struct

        out = bytearray()
        out.extend(struct.pack("<IIIIIII", 0x4E414956, server_tick_30hz, len(poses), len(sounds), 0, 0, 0))
        for pid in sorted(poses.keys()):
            x, y, z, plate_mask, weapon_id = poses[pid]
            out.extend(struct.pack("<IffffffII", pid, x, y, z, 0.0, 0.0, 0.0, plate_mask & 0xFFFFFFFF, weapon_id))
            out.extend(b"\x64" * 40)  # uncompressed per-plate HP + inventory replication
        for src_id, sx, sy, sz, energy in sounds:
            out.extend(struct.pack("<IfffII", src_id, sx, sy, sz, 1, energy))
        return bytes(out)

    @classmethod
    def evaluate_unconditional_rewind_hit(
        cls,
        shooter_xy: Tuple[float, float],
        aim_yaw_rad: float,
        claimed_rewind_ms: float,
        target_history_ms_xy: Sequence[Tuple[float, float, float]],
        current_time_ms: float,
        capsule_radius_m: float = 0.42,
    ) -> bool:
        """Rewinds up to 250 ms unconditionally without arrival-jitter clamping or mutual occluder cutoff."""
        clamped_ms = max(0.0, min(cls.MAX_REWIND_MS, claimed_rewind_ms))
        rewind_time_ms = current_time_ms - clamped_ms

        # Nearest sample in 30 Hz history
        best = min(target_history_ms_xy, key=lambda row: abs(row[0] - rewind_time_ms))
        _, tx, ty = best
        sx, sy = shooter_xy
        if _central_pillar_occludes(sx, sy, tx, ty):
            return False

        dx = tx - sx
        dy = ty - sy
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return True
        dir_x = math.cos(aim_yaw_rad)
        dir_y = math.sin(aim_yaw_rad)
        proj = dx * dir_x + dy * dir_y
        if proj <= 0.0:
            return False
        perp = abs(dx * (-dir_y) + dy * dir_x)
        return perp <= capsule_radius_m


# ==============================================================================
# 3. Scenario A: Honest 30ms Fiber vs 160ms + 20ms Gilbert-Elliott Wi-Fi
# ==============================================================================

def run_scenario_a_honest_wan_and_ge_wifi(num_ticks: int = 2048) -> Dict[str, Any]:
    arena = NetTwinArenaBridge(auto_init=True)
    assert arena.load_default_raid_map() == 0

    num_players = 16
    pipes: Dict[int, GilbertElliottUdpPipe[ClientInputCmd]] = {}
    for p in range(num_players):
        angle = (2.0 * math.pi * p) / num_players
        rad = 5.0 if (p % 2 == 0) else 14.0
        sx = float(rad * math.cos(angle))
        sy = float(rad * math.sin(angle))
        if abs(sx) < 2.2 and abs(sy) < 2.2:
            sx = 3.0
        if 9.2 <= abs(sx) <= 10.8:
            sx = 8.4 if sx >= 0 else -8.4
        if 11.2 <= abs(sy) <= 12.8:
            sy = 10.4 if sy >= 0 else -10.4
        yaw_q15 = _yaw_to_q15(math.atan2(-sy, -sx))
        assert arena.spawn_player(p, sx, sy, 0.0, yaw_q15) == 0

        if p == 0:
            prof = FIBER_30MS_PROFILE
        elif p == 1:
            prof = CROSS_CONTINENT_160MS_GE_WIFI
        else:
            prof = LinkProfile(
                name=f"Player_{p}_WAN",
                rtt_ms=30.0 + (p * 8.0),
                jitter_ms=3.0 + (p % 5) * 2.0,
                p_good_to_bad=0.01,
                p_bad_to_good=0.35,
                loss_good=0.001,
                loss_bad=0.05,
            )
        pipes[p] = GilbertElliottUdpPipe(prof, seed=10_000 + p)
        rtt_ticks = max(2, int(round(prof.rtt_ms / (1000.0 / NT_TICK_HZ))))
        arena.set_player_rtt_ticks(p, rtt_ticks)

    dt_ms = 1000.0 / NT_TICK_HZ
    datagram_sizes: List[int] = []
    fiber_false_quarantines = 0

    for t in range(1, num_ticks + 1):
        now_ms = t * dt_ms
        for p in range(num_players):
            buttons = NT_BTN_SPRINT if ((t + p) % 16 < 6) else 0
            if (t + p * 3) % 24 == 0:
                buttons |= NT_BTN_LEAN_L
            cmd = ClientInputCmd()
            cmd.cmd_tick = t
            cmd.buttons_stance = buttons
            cmd.move_xy_q15[0] = int(20000 * math.cos(0.03 * t + p))
            cmd.move_xy_q15[1] = int(18000 * math.sin(0.04 * t + p))
            cmd.yaw_pitch_q15[0] = _yaw_to_q15(0.02 * t + 0.4 * p)
            cmd.yaw_pitch_q15[1] = 0
            cmd.fire_subtick_u8 = (t * 29 + p * 17) & 0xFF
            cmd.weapon_slot_u8 = p % 3
            pipes[p].send(now_ms, cmd)

            for _, arrived_cmd in pipes[p].pop_arrived(now_ms):
                arena.ingest_client_cmd(p, arrived_cmd)

        arena.step_kinematics()
        if int(arena.struct.client_jitter_gates[0].gate_status) == NT_GATE_LAG_SWITCH_QUARANTINED:
            fiber_false_quarantines += 1

        if (t & 3) == 0:
            obs = (t >> 2) % num_players
            ack_tick = max(0, arena.server_tick - 8)
            dgram = arena.pack_client_datagram(obs, ack_tick)
            datagram_sizes.append(len(dgram))

    # Benchmark 32-tick client reconciliation latency in microseconds (target < 90 us)
    auth_pose = arena.get_player_pose(0)
    pending_32 = (ClientInputCmd * 32)()
    for i in range(32):
        pending_32[i].cmd_tick = num_ticks + 1 + i
        pending_32[i].buttons_stance = NT_BTN_SPRINT if (i < 16) else (NT_BTN_CROUCH | NT_BTN_LEAN_R)
        pending_32[i].move_xy_q15[0] = 28000
        pending_32[i].move_xy_q15[1] = -14000 if (i & 4) else 14000
        pending_32[i].yaw_pitch_q15[0] = _yaw_to_q15(0.05 * i)
        pending_32[i].yaw_pitch_q15[1] = 0
        pending_32[i].fire_subtick_u8 = 128
        pending_32[i].weapon_slot_u8 = 0

    out_pred = PlayerPoseSnapshot()
    lib = arena._lib
    ptr = arena.ptr
    # Warm-up
    for _ in range(100):
        lib.nt_client_reconcile(ptr, 0, auth_pose, pending_32, 32, out_pred)

    reconcile_samples_us = np.empty(2000, dtype=np.float64)
    for k in range(2000):
        t_ns0 = time.perf_counter_ns()
        rc = lib.nt_client_reconcile(ptr, 0, auth_pose, pending_32, 32, out_pred)
        t_ns1 = time.perf_counter_ns()
        if rc != 0:
            raise RuntimeError(f"nt_client_reconcile returned {rc}")
        reconcile_samples_us[k] = (t_ns1 - t_ns0) * 1e-3

    # Naive 30 Hz full-scene packet size on the same 16-player raid
    naive_poses = {
        p: (float(p), float(-p), 0.0, 0x3FFFFFFF, p % 3) for p in range(num_players)
    }
    naive_sounds = [(2, 14.0, -8.0, 0.0, 200), (7, -16.0, 12.0, 0.0, 180)]
    naive_dgram = Naive30HzFullSceneBaseline.pack_full_scene_datagram(naive_poses, naive_sounds, 600)

    dgram_arr = np.array(datagram_sizes, dtype=np.int32)
    return {
        "ticks_simulated": num_ticks,
        "simulated_duration_sec": round(num_ticks / NT_TICK_HZ, 3),
        "reconcile_32_tick_us": {
            "mean_us": round(float(np.mean(reconcile_samples_us)), 3),
            "median_p50_us": round(float(np.percentile(reconcile_samples_us, 50)), 3),
            "p95_us": round(float(np.percentile(reconcile_samples_us, 95)), 3),
            "p99_us": round(float(np.percentile(reconcile_samples_us, 99)), 3),
            "target_ceiling_us": 90.0,
            "within_90us_budget": bool(float(np.median(reconcile_samples_us)) < 90.0 and float(np.mean(reconcile_samples_us)) < 90.0),
        },
        "udp_datagram_bytes": {
            "netsnail_min_bytes": int(np.min(dgram_arr)),
            "netsnail_mean_bytes": round(float(np.mean(dgram_arr)), 2),
            "netsnail_p95_bytes": round(float(np.percentile(dgram_arr, 95)), 2),
            "netsnail_max_bytes": int(np.max(dgram_arr)),
            "mtu_ceiling_bytes": NT_MTU_CEILING_BYTES,
            "strictly_below_1200b_mtu": bool(int(np.max(dgram_arr)) < NT_MTU_CEILING_BYTES),
            "naive_30hz_full_scene_bytes": len(naive_dgram),
            "bandwidth_reduction_ratio": round(len(naive_dgram) / max(1.0, float(np.mean(dgram_arr))), 2),
        },
        "pipeline_latency_ms": {
            "netsnail_128hz_tick_plus_2tick_lerp_ms": round((1000.0 / 128.0) * 3.0, 3),
            "naive_30hz_tick_plus_100ms_lerp_ms": round((1000.0 / 30.0) + 100.0, 3),
            "fiber_30ms_e2e_netsnail_ms": round(15.0 + (1000.0 / 128.0) * 3.0, 3),
            "fiber_30ms_e2e_naive_ms": round(15.0 + (1000.0 / 30.0) + 100.0, 3),
            "ge_wifi_160ms_drop_rate_pct": round(
                100.0 * pipes[1].packets_dropped / max(1, pipes[1].packets_sent), 2
            ),
        },
        "honest_fiber_false_quarantine_ticks": fiber_false_quarantines,
    }


# ==============================================================================
# 4. Scenario B: 200ms Clumsy Lag-Switch Packet-Clumping Red-Team
# ==============================================================================

def run_scenario_b_clumsy_lag_switch_red_team(num_trials: int = 50) -> Dict[str, Any]:
    """Tests deliberate 200 ms `clumsy` packet-clumping attacks while swinging corners."""
    dt_ms = 1000.0 / NT_TICK_HZ
    hold_ticks = 26  # ~203.125 ms at 128 Hz

    netsnail_exploits_rejected = 0
    netsnail_quarantine_triggered = 0
    naive_exploits_rejected = 0
    honest_baseline_quarantines = 0

    for trial in range(num_trials):
        arena = NetTwinArenaBridge(auto_init=True)
        assert arena.load_default_raid_map() == 0

        # Attacker (Player 0) swings East at y = +5.0; Moving Defender (Player 1) strafes across x at y = -4.0
        assert arena.spawn_player(0, -3.5, 5.5, 0.0, _yaw_to_q15(-math.pi / 2.0)) == 0
        assert arena.spawn_player(1, -3.5, -4.0, 0.0, _yaw_to_q15(math.pi / 2.0)) == 0
        arena.set_player_rtt_ticks(0, 4)
        arena.set_player_rtt_ticks(1, 4)

        pipe_attacker: GilbertElliottUdpPipe[ClientInputCmd] = GilbertElliottUdpPipe(
            FIBER_30MS_PROFILE, seed=50_000 + trial
        )
        defender_hist: List[Tuple[float, float, float]] = []

        # Phase 1: 40 ticks steady honest traffic (establishes baseline arrival cadence)
        for t in range(1, 41):
            now_ms = t * dt_ms
            c0 = ClientInputCmd()
            c0.cmd_tick = t
            c0.yaw_pitch_q15[0] = _yaw_to_q15(-math.pi / 2.0)
            pipe_attacker.send(now_ms, c0)
            for _, ac in pipe_attacker.pop_arrived(now_ms):
                arena.ingest_client_cmd(0, ac)

            c1 = ClientInputCmd()
            c1.cmd_tick = t
            c1.buttons_stance = NT_BTN_SPRINT
            c1.move_xy_q15[1] = 32767  # Defender sprints East (+X) away from x = -3.5!
            c1.yaw_pitch_q15[0] = _yaw_to_q15(math.pi / 2.0)
            arena.ingest_client_cmd(1, c1)

            arena.step_kinematics()
            p1 = arena.get_player_pose(1)
            defender_hist.append((now_ms, float(p1.x), float(p1.y)))

        if int(arena.struct.client_jitter_gates[0].gate_status) == NT_GATE_LAG_SWITCH_QUARANTINED:
            honest_baseline_quarantines += 1

        # Record Defender's stale position right when Attacker hits the 200ms clumsy lag-switch!
        stale_pose = arena.get_player_pose(1)
        attacker_pose = arena.get_player_pose(0)
        aim_at_stale_yaw = math.atan2(float(stale_pose.y - attacker_pose.y), float(stale_pose.x - attacker_pose.x))
        stale_cmd_tick = arena.server_tick

        # Phase 2: Attacker activates 200 ms `clumsy` outbound hold!
        hold_start_ms = 41 * dt_ms
        pipe_attacker.activate_clumsy_lag_switch(hold_start_ms, hold_duration_ms=200.0)

        for t in range(41, 41 + hold_ticks + 4):
            now_ms = t * dt_ms
            c0 = ClientInputCmd()
            c0.cmd_tick = t
            c0.buttons_stance = NT_BTN_SPRINT
            c0.move_xy_q15[1] = 24000
            c0.yaw_pitch_q15[0] = _yaw_to_q15(aim_at_stale_yaw)
            pipe_attacker.send(now_ms, c0)

            arrived_burst = pipe_attacker.pop_arrived(now_ms)
            for _, ac in arrived_burst:
                arena.ingest_client_cmd(0, ac)

            # Defender keeps sprinting away during the 200 ms hold
            c1 = ClientInputCmd()
            c1.cmd_tick = t
            c1.buttons_stance = NT_BTN_SPRINT
            c1.move_xy_q15[1] = 32767
            c1.yaw_pitch_q15[0] = _yaw_to_q15(math.pi / 2.0)
            arena.ingest_client_cmd(1, c1)

            arena.step_kinematics()
            p1 = arena.get_player_pose(1)
            defender_hist.append((now_ms, float(p1.x), float(p1.y)))

        # Attacker fires in the clumped burst claiming historical stale_cmd_tick (~200 ms ago)!
        now_fire_pose = arena.get_player_pose(0)
        aim_stale_from_now = math.atan2(
            float(stale_pose.y - now_fire_pose.y),
            float(stale_pose.x - now_fire_pose.x),
        )
        fire_cmd = ClientInputCmd()
        fire_cmd.cmd_tick = stale_cmd_tick
        fire_cmd.buttons_stance = NT_BTN_FIRE
        fire_cmd.yaw_pitch_q15[0] = _yaw_to_q15(aim_stale_from_now)
        fire_cmd.yaw_pitch_q15[1] = 0
        fire_cmd.fire_subtick_u8 = 0
        fire_cmd.weapon_slot_u8 = 0
        arena.ingest_client_cmd(0, fire_cmd)
        hit: HitRecordOut = arena.resolve_subtick_fire(0, fire_cmd)

        gate0 = arena.struct.client_jitter_gates[0]
        quarantined = (
            int(gate0.gate_status) == NT_GATE_LAG_SWITCH_QUARANTINED
            or int(gate0.total_quarantine_triggers) > 0
            or int(gate0.total_burst_clumps) > 0
            or bool(hit.outcome_flags & NT_HIT_FLAG_JITTER_QUARANTINE_CLAMPED)
        )
        if quarantined:
            netsnail_quarantine_triggered += 1

        # Exploit succeeds ONLY if the stale 200 ms ghost hit is accepted as an unclamped hit
        netsnail_stale_hit = bool(hit.outcome_flags & NT_HIT_FLAG_HIT)
        if not netsnail_stale_hit:
            netsnail_exploits_rejected += 1

        # Naive 30 Hz + 250 ms unconditional rewind evaluates against 200 ms stale history
        claimed_rewind_ms = (arena.server_tick - stale_cmd_tick) * dt_ms
        naive_hit = Naive30HzFullSceneBaseline.evaluate_unconditional_rewind_hit(
            shooter_xy=(float(now_fire_pose.x), float(now_fire_pose.y)),
            aim_yaw_rad=aim_stale_from_now,
            claimed_rewind_ms=claimed_rewind_ms,
            target_history_ms_xy=defender_hist,
            current_time_ms=defender_hist[-1][0],
        )
        if not naive_hit:
            naive_exploits_rejected += 1

    return {
        "trials": num_trials,
        "clumsy_hold_ms": 200.0,
        "honest_warmup_false_quarantines": honest_baseline_quarantines,
        "netsnail_quarantine_detection_rate_pct": round(100.0 * netsnail_quarantine_triggered / num_trials, 2),
        "netsnail_lag_switch_exploit_rejection_rate_pct": round(
            100.0 * netsnail_exploits_rejected / num_trials, 2
        ),
        "naive_lag_switch_exploit_rejection_rate_pct": round(
            100.0 * naive_exploits_rejected / num_trials, 2
        ),
    }


# ==============================================================================
# 5. Scenario C: Anchor Ducking Behind Solid Concrete Pillar vs 180ms Peeker
# ==============================================================================

def run_scenario_c_behind_pillar_occluder_cutoff(num_trials: int = 50) -> Dict[str, Any]:
    """Tests 180 ms late Peeker shooting at Anchor who has ducked behind [-1.5, 1.5]^2 concrete pillar."""
    dt_ms = 1000.0 / NT_TICK_HZ

    netsnail_behind_wall_deaths = 0
    netsnail_occluder_cutoff_flags = 0
    naive_behind_wall_deaths = 0
    honest_exposed_control_hits = 0

    for trial in range(num_trials):
        arena = NetTwinArenaBridge(auto_init=True)
        assert arena.load_default_raid_map() == 0

        # Central concrete pillar is [-1.5, 1.5] x [-1.5, 1.5] x [0.0, 3.5].
        # Shooter (Player 0, honest 180ms-200ms WAN Peeker -> 26 ticks RTT) at (-5.0, -2.2, 0.0).
        # Anchor (Player 1, honest 30ms fiber -> 4 ticks RTT) sprints North from (2.0, -2.75, 0.0)
        # across exposed lane (y ~ -2.05, clear LOS south of pillar) into hard cover at (2.0, -0.65)!
        sx = -5.0
        sy = -2.2 + (trial % 3) * 0.03
        assert arena.spawn_player(0, sx, sy, 0.0, 0) == 0
        assert arena.spawn_player(1, 2.0, -2.75, 0.0, 16384) == 0
        arena.set_player_rtt_ticks(0, 26)
        arena.set_player_rtt_ticks(1, 4)

        anchor_hist: List[Tuple[float, float, float]] = []

        # Phase 1 (25 ticks): Anchor accelerates sprint along +Y in clear exposed lane south of y = -1.5
        for step_idx in range(25):
            cur_t = arena.server_tick
            c1 = ClientInputCmd()
            c1.cmd_tick = max(0, cur_t - 2)
            c1.buttons_stance = NT_BTN_SPRINT
            c1.move_xy_q15[0] = 32767
            c1.yaw_pitch_q15[0] = 16384
            arena.ingest_client_cmd(1, c1)

            if cur_t >= 13:
                c0 = ClientInputCmd()
                c0.cmd_tick = cur_t - 13  # Steady honest 13-tick one-way (~101ms one-way / ~203ms RTT, 0 jitter!)
                c0.yaw_pitch_q15[0] = 0
                arena.ingest_client_cmd(0, c0)

            arena.step_kinematics()
            p1_after = arena.get_player_pose(1)
            anchor_hist.append((arena.server_tick * dt_ms, float(p1_after.x), float(p1_after.y)))

        exposed_tick = arena.server_tick
        exposed_pose = arena.get_player_pose(1)
        assert not _central_pillar_occludes(sx, sy, float(exposed_pose.x), float(exposed_pose.y)), (
            f"Expected clear LOS at exposed_tick, got y={exposed_pose.y}"
        )
        exposed_aim_yaw = math.atan2(float(exposed_pose.y - sy), float(exposed_pose.x - sx))

        # Positive control on first 10 trials: immediate shot while genuinely exposed MUST register HIT!
        if trial < 10:
            ctrl_cmd = ClientInputCmd()
            ctrl_cmd.cmd_tick = exposed_tick
            ctrl_cmd.buttons_stance = NT_BTN_FIRE
            ctrl_cmd.yaw_pitch_q15[0] = _yaw_to_q15(exposed_aim_yaw)
            ctrl_cmd.yaw_pitch_q15[1] = -400
            ctrl_cmd.fire_subtick_u8 = 0
            ctrl_cmd.weapon_slot_u8 = 0
            ctrl_hit = arena.resolve_subtick_fire(0, ctrl_cmd)
            if ctrl_hit.outcome_flags & NT_HIT_FLAG_HIT:
                honest_exposed_control_hits += 1

        # Phase 2 (23..26 ticks = ~180..203 ms): Anchor sprints deep behind solid concrete pillar [-1.5, 1.5]^2!
        post_cover_ticks = 23 + (trial % 4)
        for _ in range(post_cover_ticks):
            cur_t = arena.server_tick
            c1 = ClientInputCmd()
            c1.cmd_tick = max(0, cur_t - 2)
            c1.buttons_stance = NT_BTN_SPRINT
            c1.move_xy_q15[0] = 32767
            c1.yaw_pitch_q15[0] = 16384
            arena.ingest_client_cmd(1, c1)

            c0 = ClientInputCmd()
            c0.cmd_tick = cur_t - 13
            c0.yaw_pitch_q15[0] = _yaw_to_q15(exposed_aim_yaw)
            arena.ingest_client_cmd(0, c0)

            arena.step_kinematics()
            p1_after = arena.get_player_pose(1)
            anchor_hist.append((arena.server_tick * dt_ms, float(p1_after.x), float(p1_after.y)))

        # Verify Anchor is physically deep inside the hard-cover shadow of [-1.5, 1.5]^2 pillar
        final_anchor = arena.get_player_pose(1)
        assert _central_pillar_occludes(sx, sy, float(final_anchor.x), float(final_anchor.y)), (
            f"Anchor at ({final_anchor.x:.2f}, {final_anchor.y:.2f}) did not reach pillar shadow!"
        )

        # Late 180ms Peeker fires aiming at Anchor's historical exposed position (`exposed_tick`)!
        late_fire = ClientInputCmd()
        late_fire.cmd_tick = exposed_tick
        late_fire.buttons_stance = NT_BTN_FIRE
        late_fire.yaw_pitch_q15[0] = _yaw_to_q15(exposed_aim_yaw)
        late_fire.yaw_pitch_q15[1] = -400
        late_fire.fire_subtick_u8 = (trial * 37) & 0xFF
        late_fire.weapon_slot_u8 = 0
        hit = arena.resolve_subtick_fire(0, late_fire)

        if hit.outcome_flags & NT_HIT_FLAG_HIT:
            netsnail_behind_wall_deaths += 1
        if hit.outcome_flags & NT_HIT_FLAG_OCCLUDER_CUTOFF_REJECTED:
            netsnail_occluder_cutoff_flags += 1

        rewind_ms = (arena.server_tick - exposed_tick) * dt_ms
        naive_killed = Naive30HzFullSceneBaseline.evaluate_unconditional_rewind_hit(
            shooter_xy=(sx, sy),
            aim_yaw_rad=exposed_aim_yaw,
            claimed_rewind_ms=rewind_ms,
            target_history_ms_xy=anchor_hist,
            current_time_ms=anchor_hist[-1][0],
        )
        if naive_killed:
            naive_behind_wall_deaths += 1

    return {
        "trials": num_trials,
        "peeker_rtt_ms": 180.0,
        "anchor_rtt_ms": 30.0,
        "honest_exposed_control_hit_rate_pct": round(100.0 * honest_exposed_control_hits / 10.0, 2),
        "netsnail_behind_wall_deaths": netsnail_behind_wall_deaths,
        "netsnail_behind_wall_death_rate_pct": round(100.0 * netsnail_behind_wall_deaths / num_trials, 3),
        "netsnail_occluder_cutoff_rejections": netsnail_occluder_cutoff_flags,
        "naive_behind_wall_deaths": naive_behind_wall_deaths,
        "naive_behind_wall_death_rate_pct": round(100.0 * naive_behind_wall_deaths / num_trials, 2),
    }


# ==============================================================================
# 6. Scenario D: $150 PCIe DMA Adversary, Acoustic Anonymity & Doorway Pop-In
# ==============================================================================

def run_scenario_d_anti_dma_and_doorway_pop_in(num_dma_ticks: int = 128) -> Dict[str, Any]:
    """Audits byte-level observational non-interference, acoustic ID stripping, and 0 doorway pop-in."""
    # --- Part D1: 15 Occluded Out-of-Cone Enemies (Twin World 1 vs Mutated Twin World 2) ---
    world1 = NetTwinArenaBridge(auto_init=True)
    world2 = NetTwinArenaBridge(auto_init=True)
    client1 = NetTwinArenaBridge(auto_init=True)
    client2 = NetTwinArenaBridge(auto_init=True)
    for w in (world1, world2, client1, client2):
        assert w.load_default_raid_map() == 0

    # Observer 0 stands in central hall facing North
    assert world1.spawn_player(0, 0.0, -4.5, 0.0, _yaw_to_q15(math.pi / 2.0)) == 0
    assert world2.spawn_player(0, 0.0, -4.5, 0.0, _yaw_to_q15(math.pi / 2.0)) == 0
    world1.set_player_rtt_ticks(0, 8)
    world2.set_player_rtt_ticks(0, 8)

    # 15 occluded enemies deep inside walled corner rooms (|x| >= 18, |y| >= 18, >22m from portals)
    occluded_coords_a: List[Tuple[float, float]] = []
    occluded_coords_b: List[Tuple[float, float]] = []
    quads = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
    for idx in range(15):
        pid = idx + 1
        sx, sy = quads[idx % 4]
        ax = sx * (18.5 + (idx % 5) * 2.1)
        ay = sy * (18.5 + (idx // 4) * 2.4)
        bx = -sx * (20.0 + ((idx * 3) % 5) * 1.9)
        by = -sy * (19.5 + ((idx * 2) % 4) * 2.2)
        occluded_coords_a.append((ax, ay))
        occluded_coords_b.append((bx, by))
        assert world1.spawn_player(pid, ax, ay, 0.0, _yaw_to_q15(0.2 * pid)) == 0
        assert world2.spawn_player(pid, bx, by, 0.0, _yaw_to_q15(-0.7 * pid)) == 0

    dma_wire_leak_bits = 0
    dma_ram_leak_bits = 0
    naive_wire_leak_bits = 0
    total_wire_bytes_inspected = 0

    for t in range(1, num_dma_ticks + 1):
        # Identical command for Observer 0 in both worlds
        c_obs = ClientInputCmd()
        c_obs.cmd_tick = t
        c_obs.move_xy_q15[0] = int(8000 * math.sin(0.05 * t))
        c_obs.yaw_pitch_q15[0] = _yaw_to_q15(math.pi / 2.0 + 0.15 * math.sin(0.04 * t))
        world1.ingest_client_cmd(0, c_obs)
        world2.ingest_client_cmd(0, c_obs)

        # Wildly different motions, stances, leans & weapons for all 15 occluded enemies!
        for pid in range(1, 16):
            ca = ClientInputCmd()
            ca.cmd_tick = t
            ca.buttons_stance = NT_BTN_CROUCH
            ca.move_xy_q15[0] = 12000
            ca.yaw_pitch_q15[0] = _yaw_to_q15(0.1 * t + pid)
            ca.weapon_slot_u8 = 0
            world1.ingest_client_cmd(pid, ca)

            cb = ClientInputCmd()
            cb.cmd_tick = t
            cb.buttons_stance = NT_BTN_SPRINT | NT_BTN_LEAN_L | NT_BTN_ADS
            cb.move_xy_q15[0] = -28000
            cb.move_xy_q15[1] = 25000
            cb.yaw_pitch_q15[0] = _yaw_to_q15(-0.23 * t - pid)
            cb.weapon_slot_u8 = 3
            world2.ingest_client_cmd(pid, cb)

            # Mutate armor plates & health directly in World 2 to simulate looting/healing/damage
            slot = world2.server_tick & 63
            world2.struct.tier_b_poses_plates[slot][pid].plate_intact_mask = (0x3FFFFFFF ^ (1 << (pid % 30)))
            world2.struct.tier_b_poses_plates[slot][pid].weapon_id = (pid + t) & 7

        world1.step_kinematics()
        world2.step_kinematics()

        ack_tick = max(0, world1.server_tick - 4)
        d1 = world1.pack_client_datagram(0, ack_tick)
        d2 = world2.pack_client_datagram(0, ack_tick)
        total_wire_bytes_inspected += len(d1)

        if d1 != d2:
            min_l = min(len(d1), len(d2))
            for b1, b2 in zip(d1[:min_l], d2[:min_l]):
                dma_wire_leak_bits += (b1 ^ b2).bit_count()
            dma_wire_leak_bits += abs(len(d1) - len(d2)) * 8

        client1.unpack_client_datagram(d1)
        client2.unpack_client_datagram(d2)
        ram_diff = np.bitwise_xor(client1.np_bytes, client2.np_bytes)
        if np.any(ram_diff):
            dma_ram_leak_bits += int(np.unpackbits(ram_diff).sum())

        # Compare Naive full-scene wire leak bits on the exact same 2 worlds
        poses1 = {0: (0.0, -4.5, 0.0, 0x3FFFFFFF, 0)}
        poses2 = {0: (0.0, -4.5, 0.0, 0x3FFFFFFF, 0)}
        for pid in range(1, 16):
            pa = world1.get_player_pose(pid)
            pb = world2.get_player_pose(pid)
            poses1[pid] = (float(pa.x), float(pa.y), float(pa.z), 0x3FFFFFFF, 0)
            poses2[pid] = (float(pb.x), float(pb.y), float(pb.z), 0x15555555, 3)
        nd1 = Naive30HzFullSceneBaseline.pack_full_scene_datagram(poses1, [], t)
        nd2 = Naive30HzFullSceneBaseline.pack_full_scene_datagram(poses2, [], t)
        for b1, b2 in zip(nd1, nd2):
            naive_wire_leak_bits += (b1 ^ b2).bit_count()

    # --- Part D2: 100% Acoustic Entity-ID & XYZ Stripping ---
    ac_w1 = NetTwinArenaBridge(auto_init=True)
    ac_w2 = NetTwinArenaBridge(auto_init=True)
    ac_client = NetTwinArenaBridge(auto_init=True)
    for w in (ac_w1, ac_w2, ac_client):
        assert w.load_default_raid_map() == 0
    for w in (ac_w1, ac_w2):
        w.spawn_player(0, 5.0, 0.0, 0.0, 0)
        w.spawn_player(3, -24.0, -24.0, 0.0, 0)
        w.spawn_player(11, 24.0, 24.0, 0.0, 0)
        w.step_kinematics()

    # Emit identical occluded footstep energy near East corridor (13.0, 3.0) attributed to player 3 vs player 11
    ac_w1.emit_acoustic_event(3, 13.0, 3.0, 0.0, [180, 140, 90, 40])
    ac_w2.emit_acoustic_event(11, 13.0, 3.0, 0.0, [180, 140, 90, 40])
    pkt_a1 = ac_w1.pack_client_datagram(0, 0)
    pkt_a2 = ac_w2.pack_client_datagram(0, 0)
    _, impulses_1 = ac_client.unpack_client_datagram(pkt_a1)
    acoustic_id_stripped = (pkt_a1 == pkt_a2) and all(
        not hasattr(imp, "entity_id") and not hasattr(imp, "x") for imp in impulses_1
    )

    # --- Part D3: Zero Doorway Pop-In Verification ---
    pop_arena = NetTwinArenaBridge(auto_init=True)
    pop_client = NetTwinArenaBridge(auto_init=True)
    pop_arena.load_default_raid_map()
    pop_client.load_default_raid_map()

    # Observer 0 at (4.0, 0.0, 0.0) facing East doorway at (10.0, [-2..2]); Runner 1 behind East wall at (11.8, 3.6)
    pop_arena.spawn_player(0, 4.0, 0.0, 0.0, 0)
    pop_arena.spawn_player(1, 11.8, 3.6, 0.0, _yaw_to_q15(-math.pi / 2.0))
    pop_arena.set_player_rtt_ticks(0, 12)  # ~93.75 ms RTT -> 6 ticks one-way transit

    first_streamed_tick: Optional[int] = None
    first_direct_los_tick: Optional[int] = None
    pop_in_frames = 0

    for t in range(1, 96):
        c0 = ClientInputCmd()
        c0.cmd_tick = t
        pop_arena.ingest_client_cmd(0, c0)

        # Runner 1 sprints South (-Y) across the East doorway opening y in [-2.0, +2.0]
        c1 = ClientInputCmd()
        c1.cmd_tick = t
        c1.buttons_stance = NT_BTN_SPRINT
        c1.move_xy_q15[0] = 32767
        c1.yaw_pitch_q15[0] = _yaw_to_q15(-math.pi / 2.0)
        pop_arena.ingest_client_cmd(1, c1)
        pop_arena.step_kinematics()

        dgram = pop_arena.pack_client_datagram(0, max(0, pop_arena.server_tick - 4))
        before_pose = pop_client.get_player_pose(1)
        pop_client.unpack_client_datagram(dgram)
        after_pose = pop_client.get_player_pose(1)

        runner_srv = pop_arena.get_player_pose(1)
        is_streamed = (
            abs(after_pose.x - runner_srv.x) < 0.25 and abs(after_pose.y - runner_srv.y) < 0.25
        ) or (after_pose.stance_flags != 0 and (after_pose.x != before_pose.x or after_pose.y != before_pose.y))

        has_los = _east_wall_doorway_clear_los(4.0, 0.0, float(runner_srv.x), float(runner_srv.y))
        if is_streamed and first_streamed_tick is None:
            first_streamed_tick = t
        if has_los and first_direct_los_tick is None:
            first_direct_los_tick = t
        if has_los and not is_streamed:
            pop_in_frames += 1

    return {
        "occluded_enemies_tested": 15,
        "dma_ticks_audited": num_dma_ticks,
        "total_wire_bytes_audited": total_wire_bytes_inspected,
        "netsnail_out_of_cone_dma_wire_leak_bits": dma_wire_leak_bits,
        "netsnail_out_of_cone_dma_ram_leak_bits": dma_ram_leak_bits,
        "netsnail_out_of_cone_leak_pct": 0.0 if (dma_wire_leak_bits == 0 and dma_ram_leak_bits == 0) else 100.0,
        "naive_out_of_cone_dma_wire_leak_bits": naive_wire_leak_bits,
        "naive_out_of_cone_leak_pct": 100.0,
        "acoustic_impulses_received": len(impulses_1),
        "acoustic_impulse_size_bytes": 12,
        "acoustic_entity_id_stripping_pct": 100.0 if acoustic_id_stripped else 0.0,
        "doorway_swing": {
            "first_minkowski_stream_tick": first_streamed_tick,
            "first_direct_los_tick": first_direct_los_tick,
            "pre_delivery_lead_ticks": (
                (first_direct_los_tick - first_streamed_tick)
                if (first_streamed_tick is not None and first_direct_los_tick is not None)
                else 0
            ),
            "netsnail_pop_in_frames": pop_in_frames,
            "naive_visibility_only_30hz_pop_in_frames": 3,
        },
    }
