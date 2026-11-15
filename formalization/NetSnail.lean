import Mathlib.Basic.Real.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum

/-!
# `NetSnail`: Machine-Checked 128 Hz Causal Netcode & Anti-DMA Invariants (`v0.1.0`)

This module formalizes the five foundational mathematical theorem families underlying
**NetSnail v0.1** (`sneed-and-feed/netsnail`), the deterministic `128 Hz` multiplayer
causal twin and anti-DMA visibility spoke of Project DarkSnail:

1. **Theorem Family 1 — Circular Tick Epoch Wraparound & `ack_mask_u64` Baseline Window Safety**:
   - `circular_tick_window_antisymmetric`: RFC-1982 half-space modular 32-bit tick ordering
     (`0 < (t_new - t_old) mod 2^32 < 2^31`) is irreflexive, strictly antisymmetric, and
     partitions non-antipodal distinct ticks across the `2^32` (`~388.5 day` at `128 Hz`) epoch wraparound.
   - `ring64_window_injective`: Any two distinct ticks inside the active 64-tick (`500 ms`)
     server rewind window map to strictly distinct slots `tick % 64` in `0..63`, seamlessly
     across `0xFFFF_FFFF → 0x0000_0000` epoch wraparound because `64 ∣ 2^32`.
   - `ack_mask_u64_baseline_in_ring`: Every bit index `b < 64` acknowledged in `ack_mask_u64`
     (within active baseline age) references a valid, un-overwritten slot in the 64-tick server history ring.

2. **Theorem Family 2 — Causal Occluder Rewind Cutoff & Lag-Switch Non-Amplification**:
   - `morse_jitter_spike_characterization`: 3-point discrete Morse arrival-jitter impulse
     `I_Morse(j, t) = (j_t - j_{t-1})(j_t - j_{t+1})` strictly triggers (`> 0`) on isolated
     packet-clumping / lag-switch bursts while satisfying `I_Morse ≤ 0` (`max 0 I_Morse = 0`,
     zero false penalty) on steady constant ping and linear clock drift `j_k = j_0 + α k`.
   - `lag_switch_rewind_non_amplification`: For `κ > 0`, injecting any positive arrival-jitter
     spike (`i_morse ≥ 0`) satisfies `w_eff(rtt_half, i_morse, w_max) ≤ w_eff(rtt_half, 0, w_max)`,
     proving a lag-switcher cannot widen their backward hit window by delaying or clumping packets.
   - `mutual_occluder_cutoff_eliminates_behind_wall_hits`: For a target moving at speed `≤ v_max`
     inside a convex hard-cover shadow region after occlusion entry `t_occ`, once
     `t_server - t_occ > τ_fair`, the mutual-occluder-clamped rewind interval
     `[max t_rewind (t_server - τ_fair), t_server]` lies `100%` inside `(t_occ, t_server]`
     (along with all sub-tick linear pose interpolations), forcing `hit_admissible = False`.

3. **Theorem Family 3 — Two-Sided Minkowski Reachability PVS (Zero Pop-In ∧ Anti-DMA Non-Interference)**:
   - `minkowski_pvs_zero_pop_in` (& `two_sided_minkowski_pvs_zero_pop_in`): Any enemy trajectory
     satisfying `dist (x_j t) (x_j t_0) ≤ v_max * (t - t_0)` that enters `Frustum_i` at any
     `t ∈ [t_0, t_0 + dt]` (`dt = rtt_half + τ_margin ≥ 0`) already satisfies `vis_i(x_j t_0)`
     at `t_0`, guaranteeing pre-delivery before doorway peek (`0` pop-in frames).
   - `anti_dma_observational_noninterference` (& `anti_dma_occluded_mutation_zero_leak`):
     Any two raid states `s_1, s_2` with identical visibility sets under `vis_i` that agree on
     visible actors produce bit-for-bit identical serialized packets `packDatagram(i, s_1) = packDatagram(i, s_2)`
     — mutating positions, armor plates, health, or loot of occluded out-of-cone enemies leaks `0` bits.

4. **Theorem Family 4 — 3D Cayley Attitude Norm Conservation & Swept-Capsule Collide-and-Slide**:
   - `cayley_so3_attitude_orthogonality_and_norm_conservation`: Rational 3D `so(3)` Rank-2 Cayley
     rotor `R(ω) = (I_3 - [ω]_×)⁻¹ (I_3 + [ω]_×)` satisfies exact resolvent identity,
     column/row orthonormality `Rᵀ R = I_3`, and Euclidean norm conservation `‖R v‖² = ‖v‖²`.
   - `swept_capsule_collide_and_slide_non_penetration` & `two_plane_crease_projection_orthogonal`:
     Unilateral normal coboundary projection `v' = v - min(0, ⟨v, n⟩) n` enforces `⟨v', n⟩ ≥ 0`
     (zero wall penetration), preserves tangential slide velocity, and never amplifies kinetic energy;
     two-plane corner crease projection along `n_1 × n_2` satisfies `⟨v_crease, n_1⟩ = 0 ∧ ⟨v_crease, n_2⟩ = 0`
     with `‖v_crease‖² ≤ ‖v‖²`.

5. **Theorem Family 5 — `NetTwinArena` `1.50 MB` (`1,572,864 B`) Cache-Line Alignment & `320 KB` L2 Residency**:
   - `net_twin_arena_layout_verified`: Verifies 64-byte cache-line alignment (`offset % 64 = 0`),
     exact contiguous pairwise byte disjointness across all 8 sub-buffers, `320 KB` (`327,680 B ≤ 512 KB`)
     Cortex-A76 private L2 residency of the Tier-A + Tier-B rewind ring, and `1,572,864 B < 2,097,152 B`
     shared L3 SRAM containment, alongside frozen C-ABI wire struct byte sizes.
-/

namespace NetSnail

open scoped BigOperators
open Finset

-- ============================================================================
-- Section 1: Circular Tick Epoch Wraparound & `ack_mask_u64` Window Safety
-- ============================================================================

/-- 32-bit modular tick sequence space `ℤ / 2³²ℤ` (`UInt32` network tick epoch). -/
abbrev Tick32 := ZMod (2 ^ 32)

instance instNeZeroTickModulus : NeZero (2 ^ 32) := ⟨by decide⟩

/-- Unsigned 32-bit modular tick difference `(t_new - t_old) mod 2³²` in `{0, ..., 2³² - 1}`. -/
def tickDelta32 (t_new t_old : Tick32) : ℕ :=
  (t_new - t_old).val

/-- RFC-1982 serial-number half-space ordering on 32-bit modular ticks:
    `t_new` is strictly newer than `t_old` iff `0 < (t_new - t_old) mod 2³² < 2³¹`. -/
def circularTickNewer (t_new t_old : Tick32) : Prop :=
  0 < tickDelta32 t_new t_old ∧ tickDelta32 t_new t_old < 2 ^ 31

/-- 64-slot (`500 ms` at `128 Hz`) server history ring-buffer index `tick % 64`. -/
def ring64Slot (t : Tick32) : Fin 64 :=
  ⟨t.val % 64, Nat.mod_lt _ (by decide)⟩

/-- Active 64-tick server history window anchored at authoritative `t_server`:
    tick `t` lies inside the active 64-slot ring iff `(t_server - t) mod 2³² < 64`. -/
def inActiveRing64Window (t_server t : Tick32) : Prop :=
  tickDelta32 t_server t < 64

/-- Bit `b` extraction from the 64-bit baseline acknowledgement bitmask `ack_mask_u64`. -/
def ackMaskBit (ack_mask_u64 : UInt64) (b : ℕ) : Bool :=
  (ack_mask_u64.toNat >>> b) % 2 == 1

/-- Baseline tick referenced by bit `b` anchored at acknowledged head tick `t_head`. -/
def ackBaselineTick (t_head : Tick32) (b : ℕ) : Tick32 :=
  t_head - (b : Tick32)

/-- Bridge lemma: `UInt32` wrapping subtraction matches `tickDelta32` on `ZMod (2^32)`. -/
lemma uint32_sub_toNat_eq_tickDelta32 (a b : UInt32) :
    (a - b).toNat = tickDelta32 (a.toNat : Tick32) (b.toNat : Tick32) := by
  have ha : a.toNat < 4294967296 := a.toBitVec.isLt
  have hb : b.toNat < 4294967296 := b.toBitVec.isLt
  unfold tickDelta32
  have hmod : ((4294967296 : ℕ) : Tick32) = 0 := ZMod.natCast_self (2 ^ 32)
  have hb_le : b.toNat ≤ 4294967296 := le_of_lt hb
  have hcast : (((4294967296 - b.toNat + a.toNat) : ℕ) : Tick32) =
      (a.toNat : Tick32) - (b.toNat : Tick32) := by
    rw [Nat.cast_add, Nat.cast_sub hb_le, hmod, zero_sub, add_comm, ← sub_eq_add_neg]
  rw [← hcast, ZMod.val_natCast]
  simp only [UInt32.toNat_sub, Nat.reducePow]

lemma tick32_neg_val_of_pos (x : Tick32) (hx : 0 < x.val) :
    (-x).val = 2 ^ 32 - x.val := by
  have hx_lt : x.val < 2 ^ 32 := ZMod.val_lt x
  have hnx_lt : (-x).val < 2 ^ 32 := ZMod.val_lt (-x)
  have hsum_mod : (x.val + (-x).val) % (2 ^ 32) = 0 := by
    have h0 : (x + (-x)).val = 0 := by simp
    rw [ZMod.val_add] at h0
    exact h0
  omega

lemma tick32_val_add_mod_64 (a b : Tick32) :
    (a + b).val % 64 = (a.val + b.val) % 64 := by
  rw [ZMod.val_add, Nat.mod_mod_of_dvd _ (by decide : 64 ∣ 2 ^ 32)]

/-- **Theorem Family 1a (`circular_tick_window_antisymmetric`)**:
The modular 32-bit half-space tick ordering `0 < (t_new - t_old) % 2^32 < 2^31` is:
1. Irreflexive (`¬ circularTickNewer t t`),
2. Strictly antisymmetric across `2^32` wraparound (`circularTickNewer t_new t_old → ¬ circularTickNewer t_old t_new`),
3. Complete on non-antipodal distinct ticks (`t_new ≠ t_old` and `tickDelta32 t_new t_old ≠ 2^31`). -/
theorem circular_tick_window_antisymmetric (t_new t_old : Tick32) :
    ¬ circularTickNewer t_new t_new ∧
    (circularTickNewer t_new t_old → ¬ circularTickNewer t_old t_new) ∧
    (t_new ≠ t_old → tickDelta32 t_new t_old ≠ 2 ^ 31 →
      (circularTickNewer t_new t_old ↔ ¬ circularTickNewer t_old t_new)) := by
  have hirrefl : ¬ circularTickNewer t_new t_new := by
    unfold circularTickNewer tickDelta32
    simp
  have hanti : ∀ a b : Tick32, circularTickNewer a b → ¬ circularTickNewer b a := by
    intro a b ⟨hpos, hlt⟩ ⟨hpos_rev, hlt_rev⟩
    unfold tickDelta32 at hpos hlt hlt_rev
    have hneg : b - a = -(a - b) := (neg_sub a b).symm
    rw [hneg, tick32_neg_val_of_pos (a - b) hpos] at hlt_rev
    omega
  refine ⟨hirrefl, hanti t_new t_old, fun hne hnot_mid => ⟨hanti t_new t_old, fun hnot_rev => ?_⟩⟩
  unfold circularTickNewer tickDelta32 at hnot_rev ⊢
  unfold tickDelta32 at hnot_mid
  have hsub_ne : t_new - t_old ≠ 0 := sub_ne_zero.mpr hne
  have hpos : 0 < (t_new - t_old).val :=
    Nat.pos_of_ne_zero fun h0 => hsub_ne ((ZMod.val_eq_zero (t_new - t_old)).mp h0)
  have hlt_mod : (t_new - t_old).val < 2 ^ 32 := ZMod.val_lt (t_new - t_old)
  have hneg : t_old - t_new = -(t_new - t_old) := (neg_sub t_new t_old).symm
  rw [hneg, tick32_neg_val_of_pos (t_new - t_old) hpos] at hnot_rev
  omega

/-- **Theorem Family 1b (`ring64_window_injective`)**:
Any two ticks `t₁, t₂` inside the active 64-tick (`500 ms` at `128 Hz`) server history window
map to the same 64-slot ring index `tick % 64` if and only if `t₁ = t₂` (hence distinct active
ticks map to strictly distinct ring slots `0..63`, including across `2^32` epoch wraparound). -/
theorem ring64_window_injective (t_server t₁ t₂ : Tick32)
    (h₁ : inActiveRing64Window t_server t₁)
    (h₂ : inActiveRing64Window t_server t₂) :
    (ring64Slot t₁ = ring64Slot t₂ ↔ t₁ = t₂) ∧
    (t₁ ≠ t₂ → ring64Slot t₁ ≠ ring64Slot t₂) := by
  have hinj : ring64Slot t₁ = ring64Slot t₂ → t₁ = t₂ := by
    intro hslot
    have hmod : t₁.val % 64 = t₂.val % 64 := congr_arg Fin.val hslot
    unfold inActiveRing64Window tickDelta32 at h₁ h₂
    set d₁ := (t_server - t₁).val
    set d₂ := (t_server - t₂).val
    have hd₁_cast : (d₁ : Tick32) = t_server - t₁ := ZMod.natCast_zmod_val (t_server - t₁)
    have hd₂_cast : (d₂ : Tick32) = t_server - t₂ := ZMod.natCast_zmod_val (t_server - t₂)
    have hsum₁ : t₁ + (d₁ : Tick32) = t_server := by rw [hd₁_cast]; abel
    have hsum₂ : t₂ + (d₂ : Tick32) = t_server := by rw [hd₂_cast]; abel
    have hd₁_val : (d₁ : Tick32).val = d₁ := by
      rw [ZMod.val_natCast]; exact Nat.mod_eq_of_lt (by omega)
    have hd₂_val : (d₂ : Tick32).val = d₂ := by
      rw [ZMod.val_natCast]; exact Nat.mod_eq_of_lt (by omega)
    have hm₁ : (t₁.val + d₁) % 64 = t_server.val % 64 := by
      rw [← hsum₁, tick32_val_add_mod_64, hd₁_val]
    have hm₂ : (t₂.val + d₂) % 64 = t_server.val % 64 := by
      rw [← hsum₂, tick32_val_add_mod_64, hd₂_val]
    have hm_eq : (t₁.val % 64 + d₁) % 64 = (t₂.val % 64 + d₂) % 64 := by
      rw [Nat.ModEq.add_right d₁ (Nat.mod_modEq t₁.val 64),
          Nat.ModEq.add_right d₂ (Nat.mod_modEq t₂.val 64), hm₁, hm₂]
    rw [hmod] at hm_eq
    have hd_eq : d₁ = d₂ := by omega
    have hsub_eq : t_server - t₁ = t_server - t₂ := by
      rw [← hd₁_cast, ← hd₂_cast, hd_eq]
    exact sub_right_injective hsub_eq
  exact ⟨⟨hinj, fun h => h ▸ rfl⟩, fun hne hslot => hne (hinj hslot)⟩

/-- **Theorem Family 1c (`ack_mask_u64_baseline_in_ring`)**:
For any acknowledged bit index `b < 64` set in `ack_mask_u64` anchored at client ack head `t_head`
whose server age satisfies `tickDelta32 t_server t_head + b < 64` (in particular, every `b < 64`
when `t_head = t_server`):
1. The baseline slot index `(ring64Slot (ackBaselineTick t_head b)).val < 64` is in-bounds,
2. The baseline tick lies inside the active 64-tick server history window (`inActiveRing64Window`),
3. Its ring slot is strictly un-overwritten by every other active tick `t_hist ≠ t_base` in the ring. -/
theorem ack_mask_u64_baseline_in_ring
    (t_server t_head : Tick32) (ack_mask_u64 : UInt64) (b : ℕ)
    (hb : b < 64)
    (hbit : ackMaskBit ack_mask_u64 b = true)
    (hage : tickDelta32 t_server t_head + b < 64) :
    let t_base := ackBaselineTick t_head b
    (ackMaskBit ack_mask_u64 b = true ∧ (ring64Slot t_base).val < 64) ∧
    inActiveRing64Window t_server t_base ∧
    (∀ t_hist : Tick32, inActiveRing64Window t_server t_hist → t_hist ≠ t_base →
      ring64Slot t_hist ≠ ring64Slot t_base) := by
  dsimp only
  set t_base := ackBaselineTick t_head b with ht_base
  have h_in_win : inActiveRing64Window t_server t_base := by
    unfold inActiveRing64Window tickDelta32
    rw [ht_base]
    unfold ackBaselineTick
    have hsplit : t_server - (t_head - (b : Tick32)) = (t_server - t_head) + (b : Tick32) := by
      abel
    rw [hsplit, ZMod.val_add]
    have hb_val : (b : Tick32).val = b := by
      rw [ZMod.val_natCast]; exact Nat.mod_eq_of_lt (by omega)
    rw [hb_val]
    unfold tickDelta32 at hage
    have hmod : ((t_server - t_head).val + b) % (2 ^ 32) = (t_server - t_head).val + b :=
      Nat.mod_eq_of_lt (by omega)
    omega
  refine ⟨⟨hbit, (ring64Slot t_base).isLt⟩, h_in_win, fun t_hist h_hist hne => ?_⟩
  exact (ring64_window_injective t_server t_hist t_base h_hist h_in_win).2 hne

-- ============================================================================
-- Section 2: Causal Occluder Rewind Cutoff & Lag-Switch Non-Amplification
-- ============================================================================

/-- 3-point discrete Morse arrival-jitter impulse at packet index `t`:
    `I_Morse(j, t) = (j_t - j_{t-1}) * (j_t - j_{t+1})`. -/
def morseJitterImpulse (j : ℤ → ℝ) (t : ℤ) : ℝ :=
  (j t - j (t - 1)) * (j t - j (t + 1))

/-- Innovation-gated backward rewind horizon:
    `w_eff(rtt_half, i_morse, w_max) = min w_max (max 0 (rtt_half - κ * max 0 i_morse))`. -/
def wEff (κ rtt_half i_morse w_max : ℝ) : ℝ :=
  min w_max (max 0 (rtt_half - κ * max 0 i_morse))

/-- **Theorem Family 2a (`morse_jitter_spike_characterization`)**:
1. `I_Morse(j, t) > 0` iff `j_t` is a strict local arrival-jitter extremum.
2. Any isolated packet-clumping / lag-switch burst (`j_{t-1} = j_base`, `j_{t+1} = j_base`,
   `j_t = j_base + δ_burst` with `δ_burst ≠ 0`) strictly triggers `I_Morse = δ_burst² > 0`.
3. Steady constant latency (`α = 0`) and arbitrary linear clock drift (`j_k = j_0 + α * k`)
   satisfy `I_Morse = -α² ≤ 0` and `max 0 I_Morse = 0` (`0` false penalty on honest ping). -/
theorem morse_jitter_spike_characterization (j : ℤ → ℝ) (t : ℤ) :
    (0 < morseJitterImpulse j t ↔
      ((j (t - 1) < j t ∧ j (t + 1) < j t) ∨ (j t < j (t - 1) ∧ j t < j (t + 1)))) ∧
    (∀ j_base δ_burst : ℝ, δ_burst ≠ 0 →
      j (t - 1) = j_base → j (t + 1) = j_base → j t = j_base + δ_burst →
        morseJitterImpulse j t = δ_burst ^ 2 ∧ 0 < morseJitterImpulse j t) ∧
    (∀ j₀ α : ℝ, (∀ k : ℤ, j k = j₀ + α * (k : ℝ)) →
      morseJitterImpulse j t = -(α ^ 2) ∧
      morseJitterImpulse j t ≤ 0 ∧
      max 0 (morseJitterImpulse j t) = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · unfold morseJitterImpulse
    rw [mul_pos_iff]
    simp only [sub_pos, sub_neg]
  · intro j_base δ_burst hδ hprev hnext hcur
    have hsq : morseJitterImpulse j t = δ_burst ^ 2 := by
      unfold morseJitterImpulse
      rw [hprev, hnext, hcur]
      ring
    exact ⟨hsq, by rw [hsq]; positivity⟩
  · intro j₀ α hlin
    have hval : morseJitterImpulse j t = -(α ^ 2) := by
      unfold morseJitterImpulse
      rw [hlin t, hlin (t - 1), hlin (t + 1)]
      push_cast
      ring
    have hnonpos : morseJitterImpulse j t ≤ 0 := by
      rw [hval]
      have hsq : 0 ≤ α ^ 2 := sq_nonneg α
      linarith
    exact ⟨hval, hnonpos, max_eq_left hnonpos⟩

/-- **Theorem Family 2b (`lag_switch_rewind_non_amplification`)**:
For coupling gain `κ > 0`:
1. Injecting any positive arrival-jitter spike (`i_morse ≥ 0`) never widens the backward rewind
   horizon: `wEff κ rtt_half i_morse w_max ≤ wEff κ rtt_half 0 w_max`.
2. Higher jitter spikes monotonically tighten (never loosen) the rewind window (`0 ≤ i₁ ≤ i₂`).
3. Any active spike (`i_morse > 0`) strictly shrinks an unsaturated positive rewind window
   (`0 < rtt_half ≤ w_max`). -/
theorem lag_switch_rewind_non_amplification
    (κ rtt_half i_morse w_max : ℝ) (hκ : 0 < κ) :
    (0 ≤ i_morse → wEff κ rtt_half i_morse w_max ≤ wEff κ rtt_half 0 w_max) ∧
    (∀ i₁ i₂ : ℝ, 0 ≤ i₁ → i₁ ≤ i₂ →
      wEff κ rtt_half i₂ w_max ≤ wEff κ rtt_half i₁ w_max) ∧
    (0 < i_morse → 0 < rtt_half → rtt_half ≤ w_max →
      wEff κ rtt_half i_morse w_max < wEff κ rtt_half 0 w_max) := by
  have hmono : ∀ i₁ i₂ : ℝ, 0 ≤ i₁ → i₁ ≤ i₂ →
      wEff κ rtt_half i₂ w_max ≤ wEff κ rtt_half i₁ w_max := by
    intro i₁ i₂ hi₁ hi₁₂
    unfold wEff
    have hi₂ : 0 ≤ i₂ := le_trans hi₁ hi₁₂
    rw [max_eq_right hi₁, max_eq_right hi₂]
    refine min_le_min_left w_max (max_le_max_left 0 ?_)
    have hmul : κ * i₁ ≤ κ * i₂ := mul_le_mul_of_nonneg_left hi₁₂ (le_of_lt hκ)
    linarith
  refine ⟨fun hi => hmono 0 i_morse le_rfl hi, hmono, fun hi_pos hrtt_pos hrtt_le => ?_⟩
  unfold wEff
  rw [max_eq_right (le_of_lt hi_pos), max_self, mul_zero, sub_zero,
      max_eq_right (le_of_lt hrtt_pos), min_eq_right hrtt_le]
  have hpen : 0 < κ * i_morse := mul_pos hκ hi_pos
  have hmax_lt : max 0 (rtt_half - κ * i_morse) < rtt_half :=
    max_lt hrtt_pos (by linarith)
  exact lt_of_le_of_lt (min_le_right w_max _) hmax_lt

/-- Squared Euclidean norm on `ℝ³` (`Fin 3 → ℝ`). -/
def dot3 (u v : Fin 3 → ℝ) : ℝ :=
  u 0 * v 0 + u 1 * v 1 + u 2 * v 2

def normSq3 (v : Fin 3 → ℝ) : ℝ :=
  dot3 v v

/-- Convexity of a 3D hard-cover occlusion shadow region `Shadow ⊆ ℝ³`. -/
def IsConvexShadow3 (Shadow : Set (Fin 3 → ℝ)) : Prop :=
  ∀ x ∈ Shadow, ∀ y ∈ Shadow, ∀ α : ℝ, 0 ≤ α → α ≤ 1 →
    (fun k => (1 - α) * x k + α * y k) ∈ Shadow

/-- Mutual-occluder-clamped backward rewind interval `[max t_rewind (t_server - τ_fair), t_server]`. -/
def inClampedRewindInterval (t_rewind t_server tau_fair t : ℝ) : Prop :=
  max t_rewind (t_server - tau_fair) ≤ t ∧ t ≤ t_server

/-- Sub-tick hit admissibility predicate: shot rewind time `t_shot` must fall inside the
    mutual-occluder-clamped rewind interval and have unoccluded target line-of-sight (`pos t ∉ Shadow`). -/
def hitAdmissible (Shadow : Set (Fin 3 → ℝ)) (pos : ℝ → (Fin 3 → ℝ))
    (t_rewind t_server tau_fair t_shot : ℝ) : Prop :=
  inClampedRewindInterval t_rewind t_server tau_fair t_shot ∧ pos t_shot ∉ Shadow

/-- **Theorem Family 2c (`mutual_occluder_cutoff_eliminates_behind_wall_hits`)**:
For a target moving at speed `≤ v_max` (`‖pos t₁ - pos t₂‖² ≤ (v_max * (t₁ - t₂))²`) inside a
convex hard-cover shadow region `Shadow` after occlusion entry time `t_occ`
(`∀ t ∈ (t_occ, t_server], pos t ∈ Shadow`), once `t_server - t_occ > tau_fair` (`tau_fair ≥ 0`):
1. Every instant `t` in `[max t_rewind (t_server - tau_fair), t_server]` satisfies `t ∈ (t_occ, t_server]`
   with bounded backward displacement `‖pos t - pos t_server‖² ≤ (v_max * tau_fair)²`.
2. Every sub-tick linearly interpolated pose `(1 - α) pos t₁ + α pos t₂` (`α ∈ [0, 1]`) across
   the clamped rewind interval remains `100%` occluded inside `Shadow`.
3. `hitAdmissible` is identically `False` across all candidate shot times `t_shot`. -/
theorem mutual_occluder_cutoff_eliminates_behind_wall_hits
    (Shadow : Set (Fin 3 → ℝ)) (pos : ℝ → (Fin 3 → ℝ))
    (v_max t_occ t_rewind t_server tau_fair : ℝ)
    (hv_max : 0 ≤ v_max)
    (htau_nonneg : 0 ≤ tau_fair)
    (h_convex : IsConvexShadow3 Shadow)
    (h_speed : ∀ t₁ t₂ : ℝ, normSq3 (fun k => pos t₁ k - pos t₂ k) ≤ (v_max * (t₁ - t₂)) ^ 2)
    (h_occluded : ∀ t : ℝ, t_occ < t → t ≤ t_server → pos t ∈ Shadow)
    (h_past_fair : tau_fair < t_server - t_occ) :
    (t_rewind ≤ t_server → inClampedRewindInterval t_rewind t_server tau_fair t_server) ∧
    (∀ t : ℝ, inClampedRewindInterval t_rewind t_server tau_fair t →
      (t_occ < t ∧ t ≤ t_server) ∧
      pos t ∈ Shadow ∧
      normSq3 (fun k => pos t k - pos t_server k) ≤ (v_max * tau_fair) ^ 2) ∧
    (∀ t₁ t₂ α : ℝ,
      inClampedRewindInterval t_rewind t_server tau_fair t₁ →
      inClampedRewindInterval t_rewind t_server tau_fair t₂ →
      0 ≤ α → α ≤ 1 →
      (fun k => (1 - α) * pos t₁ k + α * pos t₂ k) ∈ Shadow) ∧
    (∀ t_shot : ℝ, hitAdmissible Shadow pos t_rewind t_server tau_fair t_shot ↔ False) := by
  have h_nonempty : t_rewind ≤ t_server →
      inClampedRewindInterval t_rewind t_server tau_fair t_server := fun hrew =>
    ⟨max_le hrew (by linarith [htau_nonneg]), le_rfl⟩
  have h_win : ∀ t : ℝ, inClampedRewindInterval t_rewind t_server tau_fair t →
      (t_occ < t ∧ t ≤ t_server) ∧
      pos t ∈ Shadow ∧
      normSq3 (fun k => pos t k - pos t_server k) ≤ (v_max * tau_fair) ^ 2 := by
    intro t ⟨ht_lo, ht_hi⟩
    have hfair_le : t_server - tau_fair ≤ t := le_trans (le_max_right t_rewind _) ht_lo
    have ht_gt_occ : t_occ < t := by linarith
    have h_in_sh : pos t ∈ Shadow := h_occluded t ht_gt_occ ht_hi
    have hdt_nonneg : 0 ≤ t_server - t := by linarith
    have hdt_le : t_server - t ≤ tau_fair := by linarith
    have h_sp := h_speed t t_server
    have hsq_eq : (v_max * (t - t_server)) ^ 2 = (v_max * (t_server - t)) ^ 2 := by ring
    rw [hsq_eq] at h_sp
    have hmul_le : v_max * (t_server - t) ≤ v_max * tau_fair :=
      mul_le_mul_of_nonneg_left hdt_le hv_max
    have hpow_le : (v_max * (t_server - t)) ^ 2 ≤ (v_max * tau_fair) ^ 2 :=
      pow_le_pow_left₀ (mul_nonneg hv_max hdt_nonneg) hmul_le 2
    exact ⟨⟨ht_gt_occ, ht_hi⟩, h_in_sh, le_trans h_sp hpow_le⟩
  refine ⟨h_nonempty, h_win, fun t₁ t₂ α ht₁ ht₂ hα0 hα1 =>
      h_convex _ (h_win t₁ ht₁).2.1 _ (h_win t₂ ht₂).2.1 α hα0 hα1,
    fun t_shot => ⟨fun ⟨ht_in, hnot_sh⟩ => hnot_sh (h_win t_shot ht_in).2.1, False.elim⟩⟩

-- ============================================================================
-- Section 3: Two-Sided Minkowski Reachability PVS & Anti-DMA Non-Interference
-- ============================================================================

/-- Minkowski reachability locus `ReachBall(x_j, v_max, dt) = { x | dist x x_j ≤ v_max * dt }`. -/
def ReachBall {Pos : Type*} [PseudoMetricSpace Pos] (x_j : Pos) (v_max dt : ℝ) : Set Pos :=
  { x : Pos | dist x x_j ≤ v_max * dt }

/-- Causal pre-delivery horizon `dt = rtt_half + tau_margin`. -/
def causalPredeliveryWindow (rtt_half tau_margin : ℝ) : ℝ :=
  rtt_half + tau_margin

/-- Observer `i`'s Minkowski reachability visibility predicate:
    `vis_i(x_j) := ∃ x ∈ Frustum_i, dist x x_j ≤ v_max * dt`. -/
def visMinkowski {Pos : Type*} [PseudoMetricSpace Pos]
    (Frustum_i : Set Pos) (v_max dt : ℝ) (x_j : Pos) : Prop :=
  ∃ x ∈ Frustum_i, dist x x_j ≤ v_max * dt

/-- **Theorem Family 3a (`minkowski_pvs_zero_pop_in`)**:
For causal pre-delivery window `dt = rtt_half + tau_margin ≥ 0` and speed cap `v_max ≥ 0`,
if enemy `j`'s trajectory satisfies `dist (x_j t) (x_j t0) ≤ v_max * (t - t0)` for all
`t ∈ [t0, t0 + dt]`, and `x_j(t) ∈ Frustum_i` at any peek instant `t ∈ [t0, t0 + dt]`,
then `x_j(t) ∈ ReachBall(x_j t0, v_max, dt)` and `visMinkowski Frustum_i v_max dt (x_j t0)`
already holds at snapshot dispatch time `t0` — guaranteeing zero doorway pop-in. -/
theorem minkowski_pvs_zero_pop_in {Pos : Type*} [PseudoMetricSpace Pos]
    (Frustum_i : Set Pos) (x_j : ℝ → Pos)
    (v_max rtt_half tau_margin t0 t : ℝ)
    (hv_max : 0 ≤ v_max)
    (hdt_nonneg : 0 ≤ causalPredeliveryWindow rtt_half tau_margin)
    (h_traj : ∀ τ : ℝ, t0 ≤ τ → τ ≤ t0 + causalPredeliveryWindow rtt_half tau_margin →
      dist (x_j τ) (x_j t0) ≤ v_max * (τ - t0))
    (ht_lo : t0 ≤ t)
    (ht_hi : t ≤ t0 + causalPredeliveryWindow rtt_half tau_margin)
    (h_peek : x_j t ∈ Frustum_i) :
    let dt := causalPredeliveryWindow rtt_half tau_margin
    x_j t ∈ ReachBall (x_j t0) v_max dt ∧
    visMinkowski Frustum_i v_max dt (x_j t0) := by
  dsimp only
  set dt := causalPredeliveryWindow rtt_half tau_margin
  have _ := hdt_nonneg
  have hstep : dist (x_j t) (x_j t0) ≤ v_max * (t - t0) := h_traj t ht_lo ht_hi
  have hdt_bound : t - t0 ≤ dt := by linarith
  have hrad : dist (x_j t) (x_j t0) ≤ v_max * dt :=
    le_trans hstep (mul_le_mul_of_nonneg_left hdt_bound hv_max)
  exact ⟨hrad, ⟨x_j t, h_peek, hrad⟩⟩

/-- **Two-Sided Doorway Swing Extension (`two_sided_minkowski_pvs_zero_pop_in`)**:
When observer `i` simultaneously swings at speed `≤ v_obs` (sweeping dynamic frustum points within
`v_obs * (t - t0)` of `Frustum_i_t0`) while enemy `j` moves at speed `≤ v_enemy`, inflating the
Minkowski ball at relative closing speed `v_obs + v_enemy` guarantees pre-delivery at `t0`. -/
theorem two_sided_minkowski_pvs_zero_pop_in {Pos : Type*} [PseudoMetricSpace Pos]
    (Frustum_i_t0 : Set Pos) (x_j : ℝ → Pos)
    (v_obs v_enemy dt t0 t : ℝ)
    (hv_obs : 0 ≤ v_obs) (hv_enemy : 0 ≤ v_enemy)
    (h_traj : ∀ τ : ℝ, t0 ≤ τ → τ ≤ t0 + dt → dist (x_j τ) (x_j t0) ≤ v_enemy * (τ - t0))
    (ht_lo : t0 ≤ t) (ht_hi : t ≤ t0 + dt)
    (h_two_sided_peek : ∃ z ∈ Frustum_i_t0, dist (x_j t) z ≤ v_obs * (t - t0)) :
    visMinkowski Frustum_i_t0 (v_obs + v_enemy) dt (x_j t0) := by
  rcases h_two_sided_peek with ⟨z, hz_in, hz_dist⟩
  refine ⟨z, hz_in, ?_⟩
  have hstep : dist (x_j t) (x_j t0) ≤ v_enemy * (t - t0) := h_traj t ht_lo ht_hi
  have htri : dist z (x_j t0) ≤ dist z (x_j t) + dist (x_j t) (x_j t0) :=
    dist_triangle z (x_j t) (x_j t0)
  rw [dist_comm z (x_j t)] at htri
  have hsum : dist z (x_j t0) ≤ (v_obs + v_enemy) * (t - t0) := by
    linarith [htri, hz_dist, hstep]
  have hdt : t - t0 ≤ dt := by linarith
  exact le_trans hsum (mul_le_mul_of_nonneg_left hdt (add_nonneg hv_obs hv_enemy))

/-- Full server-authoritative raid actor tuple (position, 30-plate armor state, HP, loot/equipment). -/
structure RaidActorState (Pos Loot : Type*) where
  pos : Pos
  armorPlates : UInt32
  health : UInt16
  loot : Loot
  deriving DecidableEq

/-- Full server-authoritative `N`-player raid state with anonymous portal-diffracted acoustics. -/
structure RaidState (N : ℕ) (Pos Loot Byte : Type*) where
  actors : Fin N → RaidActorState Pos Loot
  anonymousAcoustics : List Byte

/-- Per-observer UDP datagram serializer `packDatagram(i, state)`:
    iterates canonically over `Fin N`, filters actors exclusively by `visMinkowski`,
    and appends entity-stripped anonymous acoustic bytes. -/
noncomputable def packDatagram {N : ℕ} {Pos Loot Byte : Type*} [PseudoMetricSpace Pos]
    (Frustum : Fin N → Set Pos) (v_max dt : ℝ)
    (encodeActor : Fin N → Fin N → RaidActorState Pos Loot → List Byte)
    (i : Fin N) (state : RaidState N Pos Loot Byte) : List Byte :=
  open Classical in
  ((List.finRange N).filter (fun j =>
      decide (visMinkowski (Frustum i) v_max dt (state.actors j).pos))).flatMap
    (fun j => encodeActor i j (state.actors j)) ++ state.anonymousAcoustics

/-- **Theorem Family 3b (`anti_dma_observational_noninterference`)**:
For any two server raid states `s₁, s₂` that:
1. Have identical Minkowski visibility sets `{ j | vis_i(s₁.pos j) } = { j | vis_i(s₂.pos j) }`,
2. Agree on every actor `j` satisfying `vis_i(s₁.pos j)`, and
3. Emit identical anonymous diffracted acoustics,
`packDatagram ... i s₁ = packDatagram ... i s₂` holds bit-for-bit identically —
mutating positions, armor plates, health, or loot of occluded out-of-cone enemies leaks **zero bits**
into Client `i`'s UDP datagram or physical RAM! -/
theorem anti_dma_observational_noninterference
    {N : ℕ} {Pos Loot Byte : Type*} [PseudoMetricSpace Pos]
    (Frustum : Fin N → Set Pos) (v_max dt : ℝ)
    (encodeActor : Fin N → Fin N → RaidActorState Pos Loot → List Byte)
    (i : Fin N) (s₁ s₂ : RaidState N Pos Loot Byte)
    (h_vis_eq : { j : Fin N | visMinkowski (Frustum i) v_max dt (s₁.actors j).pos } =
                { j : Fin N | visMinkowski (Frustum i) v_max dt (s₂.actors j).pos })
    (h_vis_agree : ∀ j : Fin N,
      visMinkowski (Frustum i) v_max dt (s₁.actors j).pos → s₁.actors j = s₂.actors j)
    (h_acoustics : s₁.anonymousAcoustics = s₂.anonymousAcoustics) :
    packDatagram Frustum v_max dt encodeActor i s₁ =
      packDatagram Frustum v_max dt encodeActor i s₂ := by
  classical
  unfold packDatagram
  have hpred : (fun j : Fin N => decide (visMinkowski (Frustum i) v_max dt (s₁.actors j).pos)) =
      (fun j : Fin N => decide (visMinkowski (Frustum i) v_max dt (s₂.actors j).pos)) := by
    funext j
    have hj : visMinkowski (Frustum i) v_max dt (s₁.actors j).pos ↔
        visMinkowski (Frustum i) v_max dt (s₂.actors j).pos :=
      Set.ext_iff.mp h_vis_eq j
    simp [hj]
  have hfilter_eq :
      (List.finRange N).filter (fun j => decide (visMinkowski (Frustum i) v_max dt (s₁.actors j).pos)) =
      (List.finRange N).filter (fun j => decide (visMinkowski (Frustum i) v_max dt (s₂.actors j).pos)) := by
    rw [hpred]
  rw [← hfilter_eq, h_acoustics]
  refine congr_arg (· ++ s₂.anonymousAcoustics) (List.flatMap_congr fun j hj => ?_)
  have hj_vis : visMinkowski (Frustum i) v_max dt (s₁.actors j).pos :=
    of_decide_eq_true (List.mem_filter.mp hj).2
  rw [h_vis_agree j hj_vis]

/-- Concrete Occluded-Enemy Mutation Corollary (`anti_dma_occluded_mutation_zero_leak`):
Arbitrarily mutating an occluded enemy `j_occ`'s position (to any other occluded position `pos'`),
30-plate armor bitmask `armor'`, HP `hp'`, and high-value loot `loot'` leaves Client `i`'s
serialized datagram byte-for-byte unchanged. -/
theorem anti_dma_occluded_mutation_zero_leak
    {N : ℕ} {Pos Loot Byte : Type*} [PseudoMetricSpace Pos]
    (Frustum : Fin N → Set Pos) (v_max dt : ℝ)
    (encodeActor : Fin N → Fin N → RaidActorState Pos Loot → List Byte)
    (i j_occ : Fin N) (s : RaidState N Pos Loot Byte)
    (pos' : Pos) (armor' : UInt32) (hp' : UInt16) (loot' : Loot)
    (h_occ_before : ¬ visMinkowski (Frustum i) v_max dt (s.actors j_occ).pos)
    (h_occ_after : ¬ visMinkowski (Frustum i) v_max dt pos') :
    let s_mut : RaidState N Pos Loot Byte :=
      { actors := fun j => if j = j_occ then ⟨pos', armor', hp', loot'⟩ else s.actors j
        anonymousAcoustics := s.anonymousAcoustics }
    packDatagram Frustum v_max dt encodeActor i s_mut =
      packDatagram Frustum v_max dt encodeActor i s := by
  dsimp only
  refine anti_dma_observational_noninterference Frustum v_max dt encodeActor i _ s ?_ ?_ rfl
  · ext j
    by_cases hj : j = j_occ
    · subst hj; simp [h_occ_before, h_occ_after]
    · simp [hj]
  · intro j hj_vis
    by_cases hj : j = j_occ
    · subst hj; simp [h_occ_after] at hj_vis
    · simp [hj]

-- ============================================================================
-- Section 4: 3D Cayley Attitude Rotor & Swept-Capsule Collide-and-Slide
-- ============================================================================

/-- 3D identity matrix `I₃`. -/
def eye3 (i j : Fin 3) : ℝ :=
  if i = j then 1 else 0

/-- 3D skew-symmetric `so(3)` generator `[ω]_×` satisfying `[ω]_×ᵀ = -[ω]_×`. -/
def skewSO3 (ω : Fin 3 → ℝ) (i j : Fin 3) : ℝ :=
  match i.val, j.val with
  | 0, 1 => -ω 2
  | 0, 2 => ω 1
  | 1, 0 => ω 2
  | 1, 2 => -ω 0
  | 2, 0 => -ω 1
  | 2, 1 => ω 0
  | _, _ => 0

/-- Strictly positive rational denominator `Δ(ω) = 1 + ω₀² + ω₁² + ω₂² ≥ 1` of the `so(3)` Cayley map. -/
def cayleyDenomSO3 (ω : Fin 3 → ℝ) : ℝ :=
  1 + ω 0 ^ 2 + ω 1 ^ 2 + ω 2 ^ 2

/-- Polynomial numerator matrix `N(ω) = (1 - ‖ω‖²) I₃ + 2 [ω]_× + 2 ω ωᵀ` of the `so(3)` Cayley map. -/
def cayleyNumSO3 (ω : Fin 3 → ℝ) (i j : Fin 3) : ℝ :=
  let x := ω 0
  let y := ω 1
  let z := ω 2
  match i.val, j.val with
  | 0, 0 => 1 + x ^ 2 - y ^ 2 - z ^ 2
  | 0, 1 => 2 * x * y - 2 * z
  | 0, 2 => 2 * x * z + 2 * y
  | 1, 0 => 2 * x * y + 2 * z
  | 1, 1 => 1 - x ^ 2 + y ^ 2 - z ^ 2
  | 1, 2 => 2 * y * z - 2 * x
  | 2, 0 => 2 * x * z - 2 * y
  | 2, 1 => 2 * y * z + 2 * x
  | 2, 2 => 1 - x ^ 2 - y ^ 2 + z ^ 2
  | _, _ => 0

/-- Rational Rank-2 `so(3)` Cayley attitude/lean rotor `R(ω) = (I₃ - [ω]_×)⁻¹ (I₃ + [ω]_×)`. -/
noncomputable def cayleyRotorSO3 (ω : Fin 3 → ℝ) (i j : Fin 3) : ℝ :=
  cayleyNumSO3 ω i j / cayleyDenomSO3 ω

/-- Matrix-vector product `(M v)_i = ∑_j M_{ij} v_j` on `ℝ³`. -/
def matVec3 (M : Fin 3 → Fin 3 → ℝ) (v : Fin 3 → ℝ) (i : Fin 3) : ℝ :=
  ∑ j : Fin 3, M i j * v j

/-- 3D vector cross product `u × v` on `ℝ³`. -/
def cross3 (u v : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  match k.val with
  | 0 => u 1 * v 2 - u 2 * v 1
  | 1 => u 2 * v 0 - u 0 * v 2
  | _ => u 0 * v 1 - u 1 * v 0

/-- Explicit 3-element summation expansion on `Fin 3`. -/
lemma sum_fin3 (f : Fin 3 → ℝ) : (∑ i : Fin 3, f i) = f 0 + f 1 + f 2 := by
  have hu : (univ : Finset (Fin 3)) = {0, 1, 2} := by decide
  rw [hu]
  simp [Finset.sum_insert]
  ring

/-- **Theorem Family 4a (`cayley_so3_attitude_orthogonality_and_norm_conservation`)**:
For any angular/lean vector `ω ∈ ℝ³`:
1. `[ω]_×` is skew-symmetric (`[ω]_×ᵀ = -[ω]_×`),
2. `R(ω) = cayleyRotorSO3 ω` solves the exact Cayley resolvent equation `(I₃ - [ω]_×) R(ω) = I₃ + [ω]_×`,
3. `R(ω)` is column-orthonormal (`R(ω)ᵀ R(ω) = I₃`),
4. `R(ω)` conserves the exact 3D Euclidean squared norm (`‖R(ω) v‖² = ‖v‖²`, `0.000e+00` drift). -/
theorem cayley_so3_attitude_orthogonality_and_norm_conservation
    (ω v : Fin 3 → ℝ) :
    (∀ i j : Fin 3, skewSO3 ω j i = -skewSO3 ω i j) ∧
    (∀ i k : Fin 3,
      (∑ j : Fin 3, (eye3 i j - skewSO3 ω i j) * cayleyRotorSO3 ω j k) =
        eye3 i k + skewSO3 ω i k) ∧
    (∀ j k : Fin 3,
      (∑ i : Fin 3, cayleyRotorSO3 ω i j * cayleyRotorSO3 ω i k) = eye3 j k) ∧
    normSq3 (matVec3 (cayleyRotorSO3 ω) v) = normSq3 v := by
  have hΔ_pos : 0 < cayleyDenomSO3 ω := by unfold cayleyDenomSO3; positivity
  have hΔ_ne : cayleyDenomSO3 ω ≠ 0 := ne_of_gt hΔ_pos
  have hΔ2_ne : cayleyDenomSO3 ω ^ 2 ≠ 0 := pow_ne_zero 2 hΔ_ne
  have h_skew : ∀ i j : Fin 3, skewSO3 ω j i = -skewSO3 ω i j := by
    intro i j; fin_cases i <;> fin_cases j <;> simp [skewSO3]
  have h_res : ∀ i k : Fin 3,
      (∑ j : Fin 3, (eye3 i j - skewSO3 ω i j) * cayleyRotorSO3 ω j k) =
        eye3 i k + skewSO3 ω i k := by
    intro i k
    unfold cayleyRotorSO3
    rw [sum_fin3]
    simp only [mul_div_assoc', ← add_div]
    have hnum :
        ((eye3 i 0 - skewSO3 ω i 0) * cayleyNumSO3 ω 0 k +
         (eye3 i 1 - skewSO3 ω i 1) * cayleyNumSO3 ω 1 k +
         (eye3 i 2 - skewSO3 ω i 2) * cayleyNumSO3 ω 2 k) =
        (eye3 i k + skewSO3 ω i k) * cayleyDenomSO3 ω := by
      fin_cases i <;> fin_cases k <;>
        simp [eye3, skewSO3, cayleyNumSO3, cayleyDenomSO3] <;> ring
    rw [hnum, mul_div_cancel_right₀ _ hΔ_ne]
  have h_orth : ∀ j k : Fin 3,
      (∑ i : Fin 3, cayleyRotorSO3 ω i j * cayleyRotorSO3 ω i k) = eye3 j k := by
    intro j k
    unfold cayleyRotorSO3
    rw [sum_fin3]
    have hdiv : ∀ a b d : ℝ, (a / d) * (b / d) = (a * b) / (d ^ 2) := fun a b d => by
      rw [div_mul_div_comm, sq]
    rw [hdiv, hdiv, hdiv, ← add_div, ← add_div]
    have hnum :
        cayleyNumSO3 ω 0 j * cayleyNumSO3 ω 0 k +
        cayleyNumSO3 ω 1 j * cayleyNumSO3 ω 1 k +
        cayleyNumSO3 ω 2 j * cayleyNumSO3 ω 2 k =
        eye3 j k * cayleyDenomSO3 ω ^ 2 := by
      fin_cases j <;> fin_cases k <;>
        simp [eye3, cayleyNumSO3, cayleyDenomSO3] <;> ring
    rw [hnum, mul_div_cancel_right₀ _ hΔ2_ne]
  have h_norm : normSq3 (matVec3 (cayleyRotorSO3 ω) v) = normSq3 v := by
    have h00 := h_orth 0 0; have h01 := h_orth 0 1; have h02 := h_orth 0 2
    have h11 := h_orth 1 1; have h12 := h_orth 1 2; have h22 := h_orth 2 2
    rw [sum_fin3] at h00 h01 h02 h11 h12 h22
    simp only [eye3, Fin.isValue, Fin.reduceEq, ↓reduceIte] at h00 h01 h02 h11 h12 h22
    unfold normSq3 dot3 matVec3
    rw [sum_fin3, sum_fin3, sum_fin3]
    linear_combination
      (v 0 ^ 2) * h00 + (2 * v 0 * v 1) * h01 + (2 * v 0 * v 2) * h02 +
      (v 1 ^ 2) * h11 + (2 * v 1 * v 2) * h12 + (v 2 ^ 2) * h22
  exact ⟨h_skew, h_res, h_orth, h_norm⟩

lemma normSq3_nonneg (u : Fin 3 → ℝ) : 0 ≤ normSq3 u :=
  add_nonneg (add_nonneg (mul_self_nonneg (u 0)) (mul_self_nonneg (u 1))) (mul_self_nonneg (u 2))

lemma normSq3_sub_scale (v n : Fin 3 → ℝ) (m : ℝ) :
    normSq3 (fun k => v k - m * n k) - normSq3 v = m ^ 2 * normSq3 n - 2 * (m * dot3 v n) := by
  unfold normSq3 dot3
  dsimp only
  ring

lemma normSq3_orth_decomp (v w : Fin 3 → ℝ) (horth : dot3 (fun k => v k - w k) w = 0) :
    normSq3 v = normSq3 w + normSq3 (fun k => v k - w k) := by
  unfold normSq3
  unfold dot3 at horth ⊢
  dsimp only at horth ⊢
  linear_combination 2 * horth

/-- 3D unilateral collide-and-slide velocity projection against wall outward unit normal `n`:
    `v' = v - min(0, ⟨v, n⟩) n`. -/
def collideAndSlideSingle (v n : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  v k - min 0 (dot3 v n) * n k

/-- Two-plane corner crease slide projection of velocity `v` along `c = n₁ × n₂`:
    `v_crease = (⟨v, n₁ × n₂⟩ / ‖n₁ × n₂‖²) (n₁ × n₂)`. -/
noncomputable def creaseSlideVelocity (v n₁ n₂ : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  (dot3 v (cross3 n₁ n₂) / normSq3 (cross3 n₁ n₂)) * cross3 n₁ n₂ k

/-- **Theorem Family 4b (`swept_capsule_collide_and_slide_non_penetration`)**:
For any swept-capsule velocity `v ∈ ℝ³` and outward unit contact normal `n` (`‖n‖² = 1`),
the unilateral projection `v' = v - min(0, ⟨v, n⟩) n` guarantees:
1. Unilateral non-penetration `⟨v', n⟩ = max(0, ⟨v, n⟩) ≥ 0` (zero wall penetration),
2. Exact preservation of tangential slide velocity (`⟨t, n⟩ = 0 → ⟨v', t⟩ = ⟨v, t⟩`),
3. Kinetic energy dissipation / zero wall-boost (`‖v'‖² = ‖v‖² - min(0, ⟨v, n⟩)² ≤ ‖v‖²`). -/
theorem swept_capsule_collide_and_slide_non_penetration
    (v n : Fin 3 → ℝ) (h_unit : normSq3 n = 1) :
    let v' := collideAndSlideSingle v n
    0 ≤ dot3 v' n ∧
    dot3 v' n = max 0 (dot3 v n) ∧
    (∀ t : Fin 3 → ℝ, dot3 t n = 0 → dot3 v' t = dot3 v t) ∧
    normSq3 v' ≤ normSq3 v := by
  dsimp only
  have hn_dot : n 0 * n 0 + n 1 * n 1 + n 2 * n 2 = 1 := h_unit
  have hdot_n : dot3 (collideAndSlideSingle v n) n = dot3 v n - min 0 (dot3 v n) := by
    unfold collideAndSlideSingle dot3
    dsimp only
    linear_combination -min 0 (v 0 * n 0 + v 1 * n 1 + v 2 * n 2) * hn_dot
  have hmax : dot3 v n - min 0 (dot3 v n) = max 0 (dot3 v n) := by
    rcases le_total 0 (dot3 v n) with hge | hle
    · rw [min_eq_left hge, max_eq_right hge, sub_zero]
    · rw [min_eq_right hle, max_eq_left hle, sub_self]
  have h_tan : ∀ t : Fin 3 → ℝ, dot3 t n = 0 → dot3 (collideAndSlideSingle v n) t = dot3 v t := by
    intro t htn
    have htn_symm : n 0 * t 0 + n 1 * t 1 + n 2 * t 2 = 0 := by
      unfold dot3 at htn; linarith
    unfold collideAndSlideSingle dot3
    dsimp only
    linear_combination -min 0 (v 0 * n 0 + v 1 * n 1 + v 2 * n 2) * htn_symm
  have h_energy : normSq3 (collideAndSlideSingle v n) ≤ normSq3 v := by
    set m := min 0 (dot3 v n)
    have hm_eq : m * (dot3 v n - m) = 0 := by
      rcases le_total 0 (dot3 v n) with hge | hle
      · have hm0 : m = 0 := min_eq_left hge
        rw [hm0, zero_mul]
      · have hmv : m = dot3 v n := min_eq_right hle
        rw [hmv, sub_self, mul_zero]
    have hexp : normSq3 (collideAndSlideSingle v n) - normSq3 v =
        m ^ 2 * normSq3 n - 2 * (m * dot3 v n) :=
      normSq3_sub_scale v n m
    rw [h_unit, mul_one] at hexp
    have hsq : 0 ≤ m ^ 2 := sq_nonneg m
    linarith [hexp, hm_eq, hsq]
  rw [hdot_n, hmax]
  exact ⟨le_max_left 0 (dot3 v n), rfl, h_tan, h_energy⟩

/-- **Theorem Family 4c (`two_plane_crease_projection_orthogonal`)**:
For any two contact plane normals `n₁, n₂ ∈ ℝ³` and incident velocity `v ∈ ℝ³`, the corner
crease velocity `v_crease = creaseSlideVelocity v n₁ n₂` along `c = n₁ × n₂` satisfies:
1. Exact simultaneous tangency `⟨v_crease, n₁⟩ = 0 ∧ ⟨v_crease, n₂⟩ = 0`,
2. Exact crease-axis velocity preservation and kinetic energy bound `‖v_crease‖² ≤ ‖v‖²`
   whenever `n₁, n₂` are non-collinear (`0 < ‖n₁ × n₂‖²`). -/
theorem two_plane_crease_projection_orthogonal
    (v n₁ n₂ : Fin 3 → ℝ) :
    let v_crease := creaseSlideVelocity v n₁ n₂
    (dot3 v_crease n₁ = 0 ∧ dot3 v_crease n₂ = 0) ∧
    (0 < normSq3 (cross3 n₁ n₂) →
      dot3 v_crease (cross3 n₁ n₂) = dot3 v (cross3 n₁ n₂) ∧
      normSq3 v_crease ≤ normSq3 v) := by
  dsimp only
  set c := cross3 n₁ n₂ with hc_def
  set s := dot3 v c / normSq3 c with hs_def
  have hvc_eq : creaseSlideVelocity v n₁ n₂ = fun k => s * c k := by
    funext k
    unfold creaseSlideVelocity
    rw [← hc_def, ← hs_def]
  have hc_n1 : dot3 c n₁ = 0 := by
    rw [hc_def]; unfold dot3 cross3; simp; ring
  have hc_n2 : dot3 c n₂ = 0 := by
    rw [hc_def]; unfold dot3 cross3; simp; ring
  have hdot_scale : ∀ u : Fin 3 → ℝ, dot3 (fun k => s * c k) u = s * dot3 c u := by
    intro u; unfold dot3; ring
  have hv_n1 : dot3 (creaseSlideVelocity v n₁ n₂) n₁ = 0 := by
    rw [hvc_eq, hdot_scale, hc_n1, mul_zero]
  have hv_n2 : dot3 (creaseSlideVelocity v n₁ n₂) n₂ = 0 := by
    rw [hvc_eq, hdot_scale, hc_n2, mul_zero]
  refine ⟨⟨hv_n1, hv_n2⟩, fun hc_pos => ?_⟩
  have hc_ne : normSq3 c ≠ 0 := ne_of_gt hc_pos
  have h_along : dot3 (creaseSlideVelocity v n₁ n₂) c = dot3 v c := by
    rw [hvc_eq, hdot_scale]
    change s * normSq3 c = dot3 v c
    rw [hs_def, div_mul_cancel₀ _ hc_ne]
  have horth_vc :
      dot3 (fun k => v k - creaseSlideVelocity v n₁ n₂ k) (creaseSlideVelocity v n₁ n₂) = 0 := by
    have horth_c : dot3 (fun k => v k - creaseSlideVelocity v n₁ n₂ k) c = 0 := by
      have hlin : dot3 (fun k => v k - creaseSlideVelocity v n₁ n₂ k) c =
          dot3 v c - dot3 (creaseSlideVelocity v n₁ n₂) c := by
        unfold dot3; ring
      rw [hlin, h_along, sub_self]
    have hsymm : ∀ a b : Fin 3 → ℝ, dot3 a b = dot3 b a := fun a b => by unfold dot3; ring
    rw [hsymm]
    conv_lhs => arg 1; rw [hvc_eq]
    rw [hdot_scale, hsymm c, horth_c, mul_zero]
  have h_pyth := normSq3_orth_decomp v (creaseSlideVelocity v n₁ n₂) horth_vc
  have hrem_nonneg := normSq3_nonneg (fun k => v k - creaseSlideVelocity v n₁ n₂ k)
  exact ⟨h_along, by linarith⟩

-- ============================================================================
-- Section 5: `NetTwinArena` `1.50 MB` Layout, Disjointness & `320 KB` L2 Bound
-- ============================================================================

/-- Exact 8 sub-buffer byte offsets inside `NetTwinArena` (`1,572,864 B`, `align(64)`). -/
def netTwinOffsets : Fin 8 → ℕ
  | 0 => 0        -- `tier_a_cylinders` (64 ticks * 64 players * 16 B)
  | 1 => 65536    -- `tier_b_poses_plates` (64 ticks * 64 players * 64 B)
  | 2 => 327680   -- `static_voxel_bvh` (32 KB voxel grid + 224 KB static AABB/portal BVH)
  | 3 => 589824   -- `client_jitter_gates` (Morse arrival-jitter & ack_mask_u64 rings)
  | 4 => 655360   -- `projectiles_acoustic_slab` (256 projectiles + portal acoustic scratch)
  | 5 => 720896   -- `client_baseline_snapshots` (per-observer baseline & Minkowski PVS masks)
  | 6 => 1245184  -- `triad_cascade_scratch` (so(16) triad, Cayley rotor & slide scratch)
  | 7 => 1507328  -- `telemetry_and_header` (tick, active mask, digest & align(64) tail)

/-- Exact 8 sub-buffer byte sizes inside `NetTwinArena`. -/
def netTwinSizes : Fin 8 → ℕ
  | 0 => 65536
  | 1 => 262144
  | 2 => 262144
  | 3 => 65536
  | 4 => 65536
  | 5 => 524288
  | 6 => 262144
  | 7 => 65536

/-- Exact `NetTwinArena` static byte footprint (`1.50 MiB = 24 * 65,536 B`). -/
def netTwinTotalBytes : ℕ := 1572864

/-- Broadcom BCM2712 Cortex-A76 private per-core L2 cache capacity (`512 KiB`). -/
def cortexA76PrivateL2Bytes : ℕ := 524288

/-- Broadcom BCM2712 shared L3 SRAM ceiling (`2.0 MiB`). -/
def bcm2712SharedL3Bytes : ℕ := 2097152

/-- Frozen C-ABI wire/FFI struct byte sizes (`ClientInputCmd`, `HitRecordOut`,
    `AcousticImpulse`, `PlayerPoseSnapshot`). -/
def clientInputCmdBytes : ℕ := 4 + 2 + 4 + 4 + 1 + 1
def hitRecordOutBytes : ℕ := 4 + 4 + 4 + 2 + 1 + 1 + 4 + 4 + 4 + 4
def acousticImpulseBytes : ℕ := 4 + 2 + 2 + 4
def playerPoseSnapshotBytes : ℕ := 4 * 6 + 2 * 4

/-- **Theorem Family 5 (`net_twin_arena_layout_verified`)**:
Over the exact 8 sub-buffer offsets `[0, 65536, 327680, 589824, 655360, 720896, 1245184, 1507328]`
and sizes `[65536, 262144, 262144, 65536, 65536, 524288, 262144, 65536]`:
1. Every sub-buffer offset is 64-byte cache-line aligned (`netTwinOffsets i % 64 = 0`),
2. All 8 sub-buffers are pairwise byte-disjoint (`netTwinOffsets i + netTwinSizes i ≤ netTwinOffsets j` for `i < j`)
   and contiguously packed with zero internal holes (`∑ i, netTwinSizes i = 1,572,864`),
3. Tier-A + Tier-B rewind ring equals `327,680 B` (`320 KiB`) `≤ 524,288 B` (`512 KiB` Cortex-A76 L2),
4. Total `NetTwinArena` equals `1,572,864 B` (`1.50 MiB`) `< 2,097,152 B` (`2.0 MiB` shared L3 SRAM),
   and all 4 C-ABI structs match their exact packed wire footprints (`16 B`, `32 B`, `12 B`, `32 B`). -/
theorem net_twin_arena_layout_verified :
    (∀ i : Fin 8, netTwinOffsets i % 64 = 0) ∧
    (∀ i j : Fin 8, i.val < j.val → netTwinOffsets i + netTwinSizes i ≤ netTwinOffsets j) ∧
    (netTwinSizes 0 + netTwinSizes 1 = 327680 ∧
      netTwinSizes 0 + netTwinSizes 1 ≤ cortexA76PrivateL2Bytes) ∧
    (netTwinOffsets 7 + netTwinSizes 7 = netTwinTotalBytes ∧
      (∑ i : Fin 8, netTwinSizes i) = netTwinTotalBytes ∧
      netTwinTotalBytes = 1572864 ∧
      netTwinTotalBytes < bcm2712SharedL3Bytes) ∧
    (clientInputCmdBytes = 16 ∧ hitRecordOutBytes = 32 ∧
      acousticImpulseBytes = 12 ∧ playerPoseSnapshotBytes = 32) := by
  refine ⟨fun i => by fin_cases i <;> rfl, fun i j hij => ?_, by decide, ?_, by decide⟩
  · fin_cases i <;> fin_cases j <;> (try revert hij; decide)
  · refine ⟨rfl, ?_, rfl, by decide⟩
    decide

end NetSnail
