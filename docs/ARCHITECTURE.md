# NetSnail Architecture & Mathematical Specification

**NetSnail** (`v0.1.0`) is the deterministic 128 Hz multiplayer netcode, sub-tick multi-plate ballistics rewind, causal occluder cutoff, and anti-DMA observational non-interference spoke of [**Project DarkSnail**](https://github.com/sneed-and-feed/projectdarksnail). It maps the machine-checked algebraic, causal, and memory-layout theorems of [`formalization/NetSnail.lean`](../formalization/NetSnail.lean) onto a zero-allocation `#![no_std]` Rust Quad-SIMD memory arena (`NetTwinArena` in [`rust/netsnail_rs/src/lib.rs`](../rust/netsnail_rs/src/lib.rs)), a stealth C99 ABI header ([`include/net_twin.h`](../include/net_twin.h)), a zero-copy Python `ctypes` FFI and red-team differential harness ([`netsnail/arena.py`](../netsnail/arena.py)), and a standalone single-file 4-pane WebGL2 + live WASM32 tactical netcode workstation ([`studio/netsnail_portable_studio.html`](../studio/netsnail_portable_studio.html)).

---

## 1. Five-Layer Deterministic Netcode Pipeline

```text
[Client 128 Hz Input Stream: ClientInputCmd (16 B)] + [Gilbert-Elliott / WAN UDP Channel]
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 3 (Ingress): Causal Aperture, 3-Point Morse Jitter Gate & Clumsy Quarantine    │
│ - 32-bit circular tick window (0 < (t_new - t_old) mod 2^32 < 2^31) + ack_mask_u64   │
│ - 3-point discrete Morse arrival-skew impulse: I_Morse(j) = (j_t - j_{t-1})(j_t-j+)  │
│ - H_clock (+) H_nope periodic lag-switch ("clumsy") detector & rewind credit clamp   │
└────────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 1 (Kinematics): Cross-ISA 0-ULP Swept Capsules, Cayley Rotor & Crease Slide    │
│ - Modular Q15 octant-folded polynomial sin/cos (zero fmod, zero FMA contraction)     │
│ - Closed-form SO(3) / so(16) Cayley lean & attitude rotor (0.000e+00 norm drift)     │
│ - 4-plane primal-active collide-and-slide (<v', n> >= 0 + n_1 x n_2 crease slide)    │
│ - Deterministic 32-tick client prediction replay (nt_client_reconcile < 90 us)       │
└────────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 2 (Combat): 320 KB L2 Sub-Tick Rewind, 30-Plate Armor & Mutual Occluder Gate   │
│ - Tier-A broadphase cylinders (64 KB) + Tier-B 64 B poses & 30-plate HP (256 KB)     │
│ - Analytic sub-tick lerp at alpha = fire_subtick_u8 / 256.0 + obliquity ricochet     │
│ - Mutual Occluder Horizon Cutoff: rejects behind-cover hits past min(RTT/2, 8 ticks) │
└────────────────────────────────────┬─────────────────────────────────────────────────┘
                                     │
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Layers 4 & 5 (Egress): Minkowski Reachability PVS, Anonymous Audio & <1200 B Delta   │
│ - Reachability ball R_j(dt) = v_max * (0.5 * RTT_i + tau_margin) vs voxel frustum    │
│ - Strict observational non-interference: 0 leaked bits for occluded out-of-cone foes │
│ - 12-byte AcousticImpulse portal diffraction (zero entity_id, zero xyz coordinates)  │
│ - Per-observer acked baseline XOR/quantized bitpacker strictly within < 1200 B MTU   │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Isomorphism Between `projectdarksnail` and `netsnail`

| Tactical Netcode Subsystem | DarkSnail / `triad-core` Primitive | Formal Guarantee & Runtime Realization |
| :--- | :--- | :--- |
| **Lag-Switch & Packet-Clump Quarantine** | 3-Point Discrete Morse Impulse Gate & $\mathcal H_{\mathrm{clock}} \oplus \mathcal H_{\mathrm{nope}}$ Splitter | `morse_jitter_spike_characterization` & `lag_switch_rewind_non_amplification`: constant-ping clock drift incurs zero penalty (`I_Morse <= 0`), whereas burst clumping strictly contracts effective rewind horizon `w_eff`. |
| **Behind-Wall Death Elimination** | Causal Mutual Occluder Horizon Cutoff | `mutual_occluder_cutoff_eliminates_behind_wall_hits`: once target hard-cover occlusion exceeds $\tau_{\mathrm{fair}} = \min(\mathrm{RTT}/2, 8\text{ ticks})$, backward hit admissibility collapses identically to `False`. |
| **Anti-DMA Wallhack & Radar Immunity** | Two-Sided Minkowski Reachability PVS | `minkowski_pvs_zero_pop_in` & `anti_dma_observational_noninterference`: pre-delivers actors before doorway peek ($0$ pop-in frames) while guaranteeing byte-identical UDP payloads under arbitrary state mutations of occluded out-of-cone enemies ($0$ leaked bits). |
| **Swept-Capsule Wall Slide & Lean Attitude** | Unilateral Contact Sheaf Coboundary & Rank-2 Cayley Rotor | $\mathrm{SO}(3)$ Cayley attitude/lean update preserves Euclidean norm (`0.000e+00` drift); primal-active normal projection and $\hat n_1 \times \hat n_2$ crease slide prevent wall tunneling. |
| **Cross-ISA Replay Lockstep (`0-ULP`)** | Non-Contracted Pairwise SIMD Trees & Modular `Q15` Polynomials | Identical canonical state digests across `x86_64 AVX2`, portable `Scalar`, and `WASM32` with `MXCSR` (`FTZ=0, DAZ=0, RN`) sanitization at C-ABI entry. |

---

## 3. Static `#![no_std]` Arena Layout (`NetTwinArena = 1,572,864 B`)

To guarantee deterministic Worst-Case Execution Time (WCET) with zero heap fragmentation or allocator lock contention across dedicated game servers, embedded edge hosts (Broadcom BCM2712 Cortex-A76), and browser WebAssembly sandboxes, the entire raid state resides in a single `#[repr(C, align(64))]` structure `NetTwinArena` of exact size **`1,572,864` bytes** (`1.50 MB` = $24 \times 65,536\text{ B}$). Every partition boundary is 64-byte cache-line aligned and enforced at compile time in Rust, C, Python, and Lean 4:

| Partition Index | Field Name | Byte Offset | Byte Size | Structural Layout & Cache Role |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `tier_a_cylinders` | `0` | `65,536 B` | `64 ticks * 64 players * 16 B` swept broadphase cylinders `[x, y, z_base, rad_height]` (`64 KB`, L1d/L2 hot). |
| `1` | `tier_b_poses_plates` | `65,536` | `262,144 B` | `64 ticks * 64 players * 64 B` (`align(64)` `PlayerTickState`): root $\mathrm{SE}(3)$ pose/velocity, `q15` Cayley lean rotor, 30-zone armor HP & material bitfields (`Tier A + Tier B = 327,680 B = 320 KB` private L2 rewind ring). |
| `2` | `static_voxel_bvh` | `327,680` | `262,144 B` | `128 * 128 * 16` 1-bit hard-cover voxel/portal occupancy grid (`32,768 B`) + static AABB brush/portal BVH (`229,376 B`). |
| `3` | `client_jitter_gates` | `589,824` | `65,536 B` | `64` players 3-point discrete Morse arrival-jitter states, $\mathcal H_{\mathrm{clock}}$ `clumsy` throttle quarantine counters, and `ack_mask_u64` baseline rings. |
| `4` | `projectiles_acoustic_slab` | `655,360` | `65,536 B` | `256` finite-velocity ballistic projectiles (`256 * 64 B = 16,384 B`) + server-side portal acoustic diffraction emitters and scratch (`49,152 B`). |
| `5` | `client_baseline_snapshots` | `720,896` | `524,288 B` | Per-observer acknowledged delta-compression baseline ring and Minkowski reachability PVS visibility bitmasks (`512 KB`). |
| `6` | `triad_cascade_scratch` | `1,245,184` | `262,144 B` | Skew-symmetric $\mathfrak{so}(16)$ tie-breaker tensors, Cayley rotor scratch, `q15` polynomial constants, and 4-plane collide-and-slide workspace (`256 KB`). |
| `7` | `telemetry_and_header` | `1,507,328` | `65,536 B` | `server_tick: u32`, `active_player_mask: u64`, `simd_mode_override: u32`, canonical 64-bit state digest, telemetry counters, and `align(64)` tail padding to `1,572,864 B`. |
| **Total** | **`NetTwinArena`** | **`0 .. 1,572,864`** | **`1,572,864 B`** | **`1.50 MB` (`75.0%` of `2.0 MB` BCM2712 L3 SRAM; `0` dynamic heap allocations).** |

---

## 4. Frozen C-ABI Wire & Telemetry Structures (`include/net_twin.h`)

All wire-facing and FFI structures enforce `#[repr(C)]` layout with zero implicit compiler padding:

1. **`ClientInputCmd` (`16 B`)**:
   - `cmd_tick: u32` (bytes `0..4`): Client command tick index.
   - `buttons_stance: u16` (bytes `4..6`): Bitmask (`bit0 = fire`, `bit1 = ads`, `bit2 = crouch`, `bit3 = sprint`, `bit4 = lean_l`, `bit5 = lean_r`).
   - `move_xy_q15: [i16; 2]` (bytes `6..10`): Normalized `Q15` wish-movement vector in $[-32767, +32767]^2$.
   - `yaw_pitch_q15: [i16; 2]` (bytes `10..14`): Modular `Q15` view angles mapping $[-32768, +32767]$ onto $[-\pi, +\pi)$.
   - `fire_subtick_u8: u8` (bytes `14..15`): Sub-tick trigger fraction $\alpha = \text{fire-subtick} / 256 \in [0, 255/256]$.
   - `weapon_slot_u8: u8` (bytes `15..16`): Active weapon ballistic profile index.
2. **`HitRecordOut` (`32 B`)**:
   - `shooter_id: u32` (`0..4`), `victim_id: u32` (`4..8`), `rewind_tick: u32` (`8..12`), `plate_zone_id: u16` (`12..14`), `outcome_flags: u8` (`14..15`: `bit0 = hit`, `bit1 = penetrated`, `bit2 = ricochet`, `bit3 = occluder_cutoff_rejected`, `bit4 = jitter_quarantine_clamped`), `_pad0: u8` (`15..16`), `subtick_alpha: f32` (`16..20`), `damage_applied: f32` (`20..24`), `armor_damage: f32` (`24..28`), `hit_dist_m: f32` (`28..32`).
3. **`AcousticImpulse` (`12 B`)**:
   - `arrival_tick: u32` (`0..4`), `azimuth_q12: i16` (`4..6`), `elevation_q12: i16` (`6..8`), `band_energy_q8: [u8; 4]` (`8..12`). Contains **zero `entity_id`** and **zero world `(x, y, z)` coordinates**, neutralizing packet-sniffing acoustic ESP radars.
4. **`PlayerPoseSnapshot` (`32 B`)**:
   - `x: f32`, `y: f32`, `z: f32`, `vx: f32`, `vy: f32`, `vz: f32`, `yaw_q15: i16`, `pitch_q15: i16`, `lean_q15: i16`, `stance_flags: u16`.

---

## 5. Mathematical Formulations & Machine-Checked Proofs (`formalization/NetSnail.lean`)

### 5.1 Circular 32-Bit Tick Epoch Wraparound & `ack_mask_u64` Window Safety (Theorem Family 1)

Server and client tick indices advance modulo $2^{32}$ at $128\text{ Hz}$ ($\Delta t = 7.8125\text{ ms}$). Ordering across epoch wraparound is governed by the modular half-space predicate:

```math
t_{\mathrm{old}} \prec_{32} t_{\mathrm{new}} \iff 0 \lt \left(t_{\mathrm{new}} - t_{\mathrm{old}}\right) \bmod 2^{32} \lt 2^{31}
```

Historical poses and acknowledged snapshot baselines are indexed into 64-slot rings ($500\text{ ms}$ active window at $128\text{ Hz}$) via $s(t) = t \bmod 64$. Theorem Family 1 establishes:
- **`circular_tick_window_antisymmetric`**: Strict irreflexivity and antisymmetry of $\prec_{32}$ across the $2^{32} - 1 \to 0$ wraparound boundary.
- **`ring64_window_injective`**: Any two distinct ticks within the active 64-tick window satisfy $t_1 \bmod 64 \ne t_2 \bmod 64$.
- **`ack_mask_u64_baseline_in_ring`**: Every bit offset $b \in \lbrace 0, \dots, 63 \rbrace$ set in `ack_mask_u64` resolves to an un-overwritten snapshot slot inside the 64-tick server baseline ring.

### 5.2 Cross-ISA `0-ULP` Kinematics, Cayley Attitude Rotors & Collide-and-Slide (Theorem Family 4)

To prevent cumulative floating-point divergence between an `x86_64` AVX2 server, an ARM64 NEON edge host, and a browser `WASM32` client during 32-tick prediction replay (`nt_client_reconcile`), NetSnail bans hardware FMA contraction and transcendental `libm` calls:

1. **Modular `Q15` Octant-Folded Polynomial Trigonometry**:
   View angles `phase_q15` in `i16` map directly to unsigned modular phase $u = \mathrm{phase} \mathbin{\&} \mathrm{0xFFFF} \in [0, 65535]$, folding exact octant symmetries onto $[0, \pi/4]$ and evaluating an `f64` Horner minimax polynomial before a single IEEE-754 round-to-nearest-even conversion to `f32` (canonicalizing `-0.0f` to `+0.0f` and any `NaN` bit-pattern to `0x7fc0_0000`).
2. **Pairwise Non-Contracted SIMD Reductions & `MXCSR` Hygiene**:
   All 16-wide dot products (`nt_dot16_pairwise_f32`) evaluate the fixed balanced binary tree:

```math
\mathrm{dot16}(\mathbf a, \mathbf b) = \Bigl(\bigl((p_0 + p_1) + (p_2 + p_3)\bigr) + \bigl((p_4 + p_5) + (p_6 + p_7)\bigr)\Bigr) + \Bigl(\bigl((p_8 + p_9) + (p_{10} + p_{11})\bigr) + \bigl((p_{12} + p_{13}) + (p_{14} + p_{15})\bigr)\Bigr)
```

   where $p_k = a_k \cdot b_k$ uses non-contracted IEEE-754 `f32` multiplication, while `x86_64` C-ABI entry points sanitize and restore `MXCSR` (`FTZ = 0`, `DAZ = 0`, `RC = RN`).
3. **3D Skew-Symmetric Cayley Attitude/Lean Rotor**:
   For skew-symmetric generator $A = \mathrm{skew}(\boldsymbol{\omega}) \in \mathfrak{so}(3)$ with $\boldsymbol{\omega} \in \mathbb R^3$, the rational Cayley map:

```math
R(\boldsymbol{\omega}) = \left(I_3 - A\right)\left(I_3 + A\right)^{-1} = I_3 + \frac{2}{1 + \lVert \boldsymbol{\omega} \rVert_2^2}\left(A + A^2\right) \in \mathrm{SO}(3)
```

   satisfies orthogonality $R(\boldsymbol{\omega})^\top R(\boldsymbol{\omega}) = I_3$ and exact Euclidean norm conservation without square roots or divisions by zero:

```math
\lVert R(\boldsymbol{\omega})\mathbf v \rVert_2^2 = \lVert \mathbf v \rVert_2^2
```

4. **Primal-Active Swept-Capsule Collide-and-Slide**:
   Against a contact plane with unit outward normal $\hat n$ ($\lVert \hat n \rVert_2 = 1$), unilateral velocity projection:

```math
\mathbf v' = \mathbf v - \min\left(0,\, \langle \mathbf v, \hat n \rangle\right)\hat n
```

   guarantees non-penetration $\langle \mathbf v', \hat n \rangle \ge 0$. When two non-parallel planes $(\hat n_1, \hat n_2)$ are simultaneously active in an acute corner, projecting onto the unit crease tangent `c_hat = (n_1 x n_2) / ||n_1 x n_2||` produces crease velocity `v_crease = <v, c_hat> c_hat` satisfying simultaneous tangency:

```math
\langle \mathbf v_{\mathrm{crease}}, \hat n_1 \rangle = 0 \quad \land \quad \langle \mathbf v_{\mathrm{crease}}, \hat n_2 \rangle = 0
```

### 5.3 3-Point Morse Jitter Quarantine & Mutual Occluder Horizon Cutoff (Theorem Family 2)

Let $j_{t-1}, j_t, j_{t+1} \in \mathbb R$ denote consecutive client command arrival-time skew residuals relative to the nominal $7.8125\text{ ms}$ tick cadence. Layer 3 evaluates the 3-point discrete Morse arrival-jitter impulse:

```math
I_{\mathrm{Morse}}(j, t) = \left(j_t - j_{t-1}\right) \cdot \left(j_t - j_{t+1}\right)
```

coupled with the innovation-gated effective rewind horizon ($\kappa > 0$):

```math
w_{\mathrm{eff}}\left(\tau_{\mathrm{rtt}}, I_{\mathrm{Morse}}, w_{\max}\right) = \min\Bigl(w_{\max},\, \max\bigl(0,\, \tau_{\mathrm{rtt}} - \kappa \max(0, I_{\mathrm{Morse}})\bigr)\Bigr)
```

Theorem Family 2 proves three adversarial combat guarantees:
1. **`morse_jitter_spike_characterization`**: Constant latency or steady linear clock drift (`j_t = j_0 + delta * t`) yields $I_{\mathrm{Morse}}(j, t) = -\delta^2 \le 0$ (zero false penalty on honest high-ping links), whereas any isolated packet-clumping / lag-switch burst (`j_t > j_{t-1}` and `j_t > j_{t+1}`) strictly triggers `I_Morse > 0`.
2. **`lag_switch_rewind_non_amplification`**: For every positive jitter impulse (`I_Morse >= 0`), the gated window satisfies $w_{\mathrm{eff}}(\tau_{\mathrm{rtt}}, I_{\mathrm{Morse}}, w_{\max}) \le w_{\mathrm{eff}}(\tau_{\mathrm{rtt}}, 0, w_{\max})$, proving a `clumsy` throttle attacker can never expand their backward hit window by buffering packets.
3. **`mutual_occluder_cutoff_eliminates_behind_wall_hits`**: Suppose target $B$ enters continuous hard-cover shadow behind `static_voxel_bvh` at tick `t_occ` relative to shooter $A$'s muzzle. Layer 2 clamps backward rewind against the mutual occluder fairness bound $\tau_{\mathrm{fair}} = \min(\lfloor T_B / 2 \rfloor, 8\text{ ticks})$ ($\le 62.5\text{ ms}$ at $128\text{ Hz}$):

```math
t_{\mathrm{eval}} \in \left[\max\left(t_{\mathrm{srv}} - w_{\mathrm{eff}},\, t_{\mathrm{srv}} - \tau_{\mathrm{fair}}\right),\, t_{\mathrm{srv}}\right]
```

   Once $t_{\mathrm{srv}} - t_{\mathrm{occ}} > \tau_{\mathrm{fair}}$, every tick in the clamped evaluation interval lies strictly inside `(t_occ, t_srv]` where line-of-sight is obstructed by solid geometry, rejecting the late shot (`outcome_flags |= 0x08`, `0.0%` behind-wall deaths).

### 5.4 Two-Sided Minkowski Reachability PVS & Anti-DMA Non-Interference (Theorem Family 3)

A physical PCIe Direct Memory Access (DMA) card attached to Client $i$'s host reads Client $i$'s physical RAM passively without executing code on the client OS. The only mathematically sound defense against DMA wallhacks and radar cheats is **information-theoretic non-transmission**: bytes describing an unobservable enemy must never enter Client $i$'s network socket.

For observer $i$ with visibility frustum $\mathcal F_i \subset \mathbb R^3$ through `static_voxel_bvh` portals and lookahead horizon `dt_i = 0.5 * RTT_i + tau_margin >= 0`, Layer 4 expands each remote actor `j` at position `x_j(t_0)` with speed bound `v_max` into its Minkowski reachability ball:

```math
\mathcal R_j(\Delta t_i) = \left\lbrace \mathbf x \in \mathbb R^3 \;\middle|\; \mathrm{dist}\left(\mathbf x, \mathbf x_j(t_0)\right) \le v_{\max} \Delta t_i \right\rbrace
```

and defines the server-side PVS filter predicate $V_i$:

```math
V_i\left(\mathbf x_j(t_0)\right) \iff \exists \mathbf x \in \mathcal F_i,\; \mathrm{dist}\left(\mathbf x, \mathbf x_j(t_0)\right) \le v_{\max} \Delta t_i
```

Theorem Family 3 proves both sides of the security-playability boundary:
- **Zero Doorway Pop-In (`minkowski_pvs_zero_pop_in`)**: For any kinematically admissible trajectory satisfying $\mathrm{dist}(\mathbf x_j(t), \mathbf x_j(t_0)) \le v_{\max}(t - t_0)$ over `t in [t_0, t_0 + dt_i]`, if `x_j(t)` enters frustum `F_i` at any moment during the transit window, then `vis_i(x_j(t_0))` already holds at `t_0` — guaranteeing the server streams actor `j`'s state before they clear the doorway edge.
- **Strict Observational Non-Interference (`anti_dma_observational_noninterference`)**: For any two raid states $s_1, s_2$ whose visible actor sets coincide (`{ j | vis_i(s1.pos j) } = { j | vis_i(s2.pos j) }`) and whose states agree on every visible actor `j`, the serialized UDP datagrams satisfy:

```math
\mathrm{packDatagram}(i, s_1) = \mathrm{packDatagram}(i, s_2)
```

  identically bit-for-bit. Arbitrary mutations to positions, velocities, 30-plate armor HP, or inventory of occluded out-of-cone actors leak **zero bits** into `out_mtu_buf` or Client $i$'s RAM. Audible occluded footsteps and gunfire traverse portal graphs purely as 12-byte `AcousticImpulse` diffraction records carrying quantized `(arrival_tick, azimuth_q12, elevation_q12, band_energy_q8)` with zero entity identifiers and zero world coordinates.

### 5.5 Machine-Checked Theorem Index (`formalization/NetSnail.lean`)

All 14 theorems (`879` lines, toolchain `leanprover/lean4:v4.34.0-rc2`) compile with `0` errors, `0` warnings, `0` `sorry`, and `0` custom axioms:

| Theorem Family | Formal Symbol (`NetSnail.*`) | Line Range | Machine-Checked Guarantee |
| :---: | :--- | :---: | :--- |
| `1a` | `circular_tick_window_antisymmetric` | `L144–L167` | RFC-1982 32-bit half-space tick ordering is irreflexive, strictly antisymmetric, and complete on non-antipodal distinct ticks across $2^{32}$ wraparound. |
| `1b` | `ring64_window_injective` | `L173–L204` | Distinct ticks in the active 64-tick (`500 ms`) window map to strictly distinct ring slots `0..63` across $2^{32}$ wraparound. |
| `1c` | `ack_mask_u64_baseline_in_ring` | `L213–L240` | Every acknowledged bit $b < 64$ in `ack_mask_u64` within active window age resolves to an un-overwritten server baseline ring slot. |
| `2a` | `morse_jitter_spike_characterization` | `L262–L292` | Isolated packet-clumping bursts trigger $I_{\mathrm{Morse}} = \delta^2 > 0$, while constant ping and linear clock drift satisfy `I_Morse = -alpha^2 <= 0`. |
| `2b` | `lag_switch_rewind_non_amplification` | `L301–L324` | Positive arrival-jitter spikes monotonically contract (never widen) effective backward rewind credit `wEff`. |
| `2c` | `mutual_occluder_cutoff_eliminates_behind_wall_hits` | `L357–L400` | Once target hard-cover occlusion exceeds $\tau_{\mathrm{fair}}$, all clamped rewind ticks and sub-tick interpolations remain inside `Shadow` (`hitAdmissible = False`). |
| `3a` | `minkowski_pvs_zero_pop_in` & `two_sided_minkowski_pvs_zero_pop_in` | `L426–L469` | Any enemy trajectory entering observer frustum within `dt = rtt_half + tau_margin` already satisfies `visMinkowski` at dispatch time `t0` (`0` pop-in frames). |
| `3b` | `anti_dma_observational_noninterference` & `anti_dma_occluded_mutation_zero_leak` | `L504–L561` | Raid states agreeing on visible actors emit bit-identical `packDatagram` bytes; mutating occluded out-of-cone enemies leaks `0` bits. |
| `4a` | `cayley_so3_attitude_orthogonality_and_norm_conservation` | `L631–L686` | Rational $\mathfrak{so}(3)$ Cayley rotor solves `R(w)^T R(w) = I_3` and preserves Euclidean norm $\lVert R\mathbf v \rVert_2^2 = \lVert \mathbf v \rVert_2^2$. |
| `4b` | `swept_capsule_collide_and_slide_non_penetration` | `L720–L759` | Unilateral wall projection enforces $\langle \mathbf v', \hat n \rangle \ge 0$, preserves tangential slide, and prevents wall-boost (`||v'||^2 <= ||v||^2`). |
| `4c` | `two_plane_crease_projection_orthogonal` | `L767–L810` | Two-plane corner crease slide along `n_1 x n_2` satisfies $\langle \mathbf v_{\mathrm{crease}}, \hat n_1 \rangle = 0 \land \langle \mathbf v_{\mathrm{crease}}, \hat n_2 \rangle = 0$. |
| `5` | `net_twin_arena_layout_verified` | `L863–L877` | Verifies `% 64 = 0` alignment, pairwise disjointness across all 8 partitions, `327,680 B <= 524,288 B` L2 residency, `1,572,864 B < 2,097,152 B` L3 fit, and C-ABI struct sizes. |

---

## 6. C-ABI Surface (`include/net_twin.h` & `wasm32-unknown-unknown`)

Exported identically across native shared libraries (`netsnail_rs.dll` / `libnetsnail_rs.so` / `libnetsnail_rs.dylib`) and standalone WebAssembly (`netsnail_rs.wasm`):

- **Lifecycle, Map Geometry & Acoustic Emitters**:
  - `int32_t nt_arena_init(NetTwinArena *arena)`
  - `int32_t nt_set_simd_backend_override(NetTwinArena *arena, uint32_t mode)` (`0 = auto SIMD`, `1 = force portable scalar`)
  - `int32_t nt_load_default_raid_map(NetTwinArena *arena)`
  - `int32_t nt_spawn_player(NetTwinArena *arena, uint32_t player_id, float x, float y, float z, int16_t yaw_q15)`
  - `int32_t nt_get_player_pose(const NetTwinArena *arena, uint32_t player_id, PlayerPoseSnapshot *out_pose)`
  - `int32_t nt_set_player_rtt_ticks(NetTwinArena *arena, uint32_t player_id, uint32_t rtt_ticks)`
  - `int32_t nt_emit_acoustic_event(NetTwinArena *arena, uint32_t source_player_id, float x, float y, float z, const uint8_t band_energy_q8[4])`
- **128 Hz Ingress, Kinematics & Prediction Replay**:
  - `int32_t nt_ingest_client_cmd(NetTwinArena *arena, uint32_t player_id, const ClientInputCmd *cmd)`
  - `int32_t nt_step_kinematics(NetTwinArena *arena)`
  - `int32_t nt_client_reconcile(NetTwinArena *arena, uint32_t player_id, const PlayerPoseSnapshot *auth_pose, const ClientInputCmd *pending_cmds, uint32_t num_cmds, PlayerPoseSnapshot *out_predicted_pose)`
- **Sub-Tick Multi-Plate Rewind & Mutual Occluder Gate**:
  - `int32_t nt_resolve_subtick_fire(NetTwinArena *arena, uint32_t shooter_id, const ClientInputCmd *fire_cmd, HitRecordOut *out_hit)`
- **Minkowski PVS Delta Bitpacker & Canonical State Digest**:
  - `int32_t nt_pack_client_datagram(NetTwinArena *arena, uint32_t observer_id, uint32_t baseline_ack_tick, uint8_t *out_mtu_buf, uint32_t max_buf_len)`
  - `int32_t nt_unpack_client_datagram(NetTwinArena *client_arena, const uint8_t *in_mtu_buf, uint32_t buf_len, AcousticImpulse *out_acoustics, uint32_t max_acoustics, uint32_t *out_num_acoustics)`
  - `uint64_t nt_hash_arena_deterministic(const NetTwinArena *arena)`
- **Tier-0 Cross-ISA Primitives & Zero-Alloc WASM32 Static Buffers**:
  - `nt_sin_q15`, `nt_cos_q15`, `nt_atan2_f32`, `nt_sqrt_f32`, `nt_dot16_pairwise_f32`, `nt_cayley_rotor_so3_f32`
  - `nt_wasm_global_arena`, `nt_wasm_cmd_buf`, `nt_wasm_mtu_buf`, `nt_wasm_hit_buf`, `nt_wasm_acoustic_buf`

---

## 7. Empirical Verification & Hardware Receipts (`results/*.json`)

All runtime, differential chaos, red-team combat, and hardware cache metrics are reproducibly generated by [`scripts/evaluate_netsnail.py`](../scripts/evaluate_netsnail.py), [`netsnail/fuzzer.py`](../netsnail/fuzzer.py), [`scripts/verify_bcm2712_arena.py`](../scripts/verify_bcm2712_arena.py), and [`scripts/verify_zero_alloc.py`](../scripts/verify_zero_alloc.py):

| Verification Dimension | Measured Receipt | Artifact Source |
| :--- | :--- | :--- |
| **Cross-ISA `0-ULP` Lockstep (`160,000` player-ticks)** | `16` players $\times$ `10,000` ticks (`20,000` shots, `732` hits, `7` ricochets, `648` penetrations): **`0` mismatched ticks**, `BLAKE3-256 = 7ddbb0e1b723f34ac0cfd5794c634c6bcee08a7cab1ba20a4c1cb8497c4519be` identically on `x86_64 SIMD`, `Scalar`, and `WASM32`; `0.000e+00` Cayley drift. | [`results/cross_isa_blake3_receipts.json`](../results/cross_isa_blake3_receipts.json) |
| **Hostile `MXCSR` (`FTZ/DAZ`) Caller Resilience** | `8` players $\times$ `512` ticks (`4,096` player-ticks) under `_DN_FLUSH`: **`0` tick mismatches** (`aeb174a5...e24922f`), caller `MXCSR` control word preserved. | [`results/cross_isa_blake3_receipts.json`](../results/cross_isa_blake3_receipts.json) |
| **Broadcom BCM2712 Cortex-A76 Cache (`50M` refs)** | `L1d = 72.15%`, `L2 = 98.57%` (`5/8` ways), `L3 = 97.42%` (`12/16` ways), **`0` L3 capacity evictions**, **`LLd miss rate = 0.01029% < 0.100%`**. | [`results/bcm2712_cache_receipts.json`](../results/bcm2712_cache_receipts.json) |
| **Static Zero-Alloc Binary Import Audit** | `netsnail_rs.dll` (`115,712 B`, `26` exports) & `netsnail_rs.wasm` (`99,689 B`, `29` exports, `0` WASM imports): **`0` dynamic heap allocator imports**. | [`results/bcm2712_cache_receipts.json`](../results/bcm2712_cache_receipts.json) |
| **32-Tick `nt_client_reconcile` Latency & `< 1200 B` MTU** | `19.90 us` mean (`17.60 us` p50, `36.10 us` p99 Python FFI; `8.85 us` native Rust, `< 90.0 us` budget); 16-player UDP datagram `92.61 B` mean / `168 B` max (`13.95x` smaller than `1,292 B` naive). | [`results/netsnail_evaluation_summary.json`](../results/netsnail_evaluation_summary.json) |
| **Mutual Occluder Cutoff & `clumsy` Lag-Switch Defense** | **`0.0%` behind-wall deaths** (`0 / 50` vs `100.0%` naive; `100.0%` exposed hit control); **`100.0%` `200 ms` `clumsy` burst quarantine** (`0` honest false quarantines). | [`results/netsnail_evaluation_summary.json`](../results/netsnail_evaluation_summary.json) |
| **Anti-DMA Non-Interference & Doorway PVS** | **`0` wire leak bits & `0` RAM leak bits** (`0.0%` vs `80,355` bits naive across `15` occluded enemies); **`0` doorway pop-in frames** (`12`-tick pre-delivery lead); **`100.0%` acoustic ID stripping**. | [`results/netsnail_evaluation_summary.json`](../results/netsnail_evaluation_summary.json) |
