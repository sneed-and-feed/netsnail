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
# `NetSnail`: Machine-Checked 128 Hz Causal Netcode & Anti-DMA Invariants (`v0.2.0`)
1. **Circular Tick & `ack_mask_u64` Ring**: antisymmetry, 64-slot injectivity, baseline safety.
2. **Causal Occluder Cutoff & Morse Gate**: spike algebra, non-amplification, zero wall-bangs.
3. **Two-Sided Minkowski PVS & Anti-DMA**: zero pop-in completeness & occluded non-interference.
4. **3D `so(3)` Cayley Rotor & Collide-and-Slide**: orthogonality, norm conservation, crease slide.
5. **`NetTwinArena` `1.50 MB` Layout**: 64-B alignment, disjointness, `320 KB` L2 residency.
6. **`v0.2` Delta Chains & 30-Plate Armor**: exact modular delta telescoping & plate monotonicity.
-/
namespace NetSnail
open scoped BigOperators
open Finset

/-! ### Section 1: Circular Tick Epoch Wraparound & `ack_mask_u64` Window Safety -/
abbrev Tick32 := ZMod (2 ^ 32)
instance instNeZeroTickModulus : NeZero (2 ^ 32) := ⟨by decide⟩
def tickDelta32 (t_new t_old : Tick32) : ℕ := (t_new - t_old).val
def circularTickNewer (a b : Tick32) : Prop := 0 < tickDelta32 a b ∧ tickDelta32 a b < 2 ^ 31
def ring64Slot (t : Tick32) : Fin 64 := ⟨t.val % 64, Nat.mod_lt _ (by decide)⟩
def inActiveRing64Window (t_server t : Tick32) : Prop := tickDelta32 t_server t < 64
def ackMaskBit (ack_mask_u64 : UInt64) (b : ℕ) : Bool := (ack_mask_u64.toNat >>> b) % 2 == 1
def ackBaselineTick (t_head : Tick32) (b : ℕ) : Tick32 := t_head - (b : Tick32)

lemma uint32_sub_toNat_eq_tickDelta32 (a b : UInt32) :
    (a - b).toNat = tickDelta32 (a.toNat : Tick32) (b.toNat : Tick32) := by
  have hb : b.toNat ≤ 4294967296 := le_of_lt b.toBitVec.isLt
  have h0 : ((4294967296 : ℕ) : Tick32) = 0 := ZMod.natCast_self (2 ^ 32)
  have hc : (((4294967296 - b.toNat + a.toNat) : ℕ) : Tick32) = (a.toNat : Tick32) - b.toNat := by
    rw [Nat.cast_add, Nat.cast_sub hb, h0, zero_sub, add_comm, ← sub_eq_add_neg]
  unfold tickDelta32; rw [← hc, ZMod.val_natCast]; simp only [UInt32.toNat_sub, Nat.reducePow]
lemma tick32_neg_val_of_pos (x : Tick32) (hx : 0 < x.val) : (-x).val = 2 ^ 32 - x.val := by
  have h0 : (x + (-x)).val = 0 := by simp
  rw [ZMod.val_add] at h0; have := ZMod.val_lt x; have := ZMod.val_lt (-x); omega

theorem circular_tick_window_antisymmetric (t_new t_old : Tick32) :
    ¬ circularTickNewer t_new t_new ∧ (circularTickNewer t_new t_old → ¬ circularTickNewer t_old t_new) ∧
    (t_new ≠ t_old → tickDelta32 t_new t_old ≠ 2 ^ 31 →
      (circularTickNewer t_new t_old ↔ ¬ circularTickNewer t_old t_new)) := by
  have hanti : ∀ a b : Tick32, circularTickNewer a b → ¬ circularTickNewer b a := by
    rintro a b ⟨hpos, hlt⟩ ⟨_, hrev⟩; unfold tickDelta32 at hpos hlt hrev
    rw [show b - a = -(a - b) from (neg_sub a b).symm, tick32_neg_val_of_pos _ hpos] at hrev; omega
  refine ⟨by simp [circularTickNewer, tickDelta32], hanti t_new t_old,
    fun hne hmid => ⟨hanti t_new t_old, fun hnot => ?_⟩⟩
  unfold circularTickNewer tickDelta32 at hnot hmid ⊢
  have hpos := Nat.pos_of_ne_zero fun h0 => sub_ne_zero.mpr hne ((ZMod.val_eq_zero _).mp h0)
  rw [show t_old - t_new = -(t_new - t_old) from (neg_sub _ _).symm, tick32_neg_val_of_pos _ hpos] at hnot
  have := ZMod.val_lt (t_new - t_old); omega
theorem ring64_window_injective (t_server t₁ t₂ : Tick32)
    (h₁ : inActiveRing64Window t_server t₁) (h₂ : inActiveRing64Window t_server t₂) :
    (ring64Slot t₁ = ring64Slot t₂ ↔ t₁ = t₂) ∧ (t₁ ≠ t₂ → ring64Slot t₁ ≠ ring64Slot t₂) := by
  have hinj : ring64Slot t₁ = ring64Slot t₂ → t₁ = t₂ := fun hslot => by
    have hmod : t₁.val % 64 = t₂.val % 64 := congr_arg Fin.val hslot
    unfold inActiveRing64Window tickDelta32 at h₁ h₂
    have hm : ∀ t : Tick32, ((t_server - t).val + t.val) % 64 = t_server.val % 64 := fun t => by
      rw [← Nat.mod_mod_of_dvd _ (by decide : 64 ∣ 2 ^ 32), ← ZMod.val_add, sub_add_cancel]
    have hm1 := hm t₁; have hm2 := hm t₂
    exact sub_right_injective (ZMod.val_injective (2 ^ 32)
      (show (t_server - t₁).val = (t_server - t₂).val by omega))
  exact ⟨⟨hinj, fun h => h ▸ rfl⟩, fun hne hslot => hne (hinj hslot)⟩
theorem ack_mask_u64_baseline_in_ring (t_server t_head : Tick32) (ack_mask_u64 : UInt64) (b : ℕ)
    (hb : b < 64) (hbit : ackMaskBit ack_mask_u64 b = true) (hage : tickDelta32 t_server t_head + b < 64) :
    let t_base := ackBaselineTick t_head b
    (ackMaskBit ack_mask_u64 b = true ∧ (ring64Slot t_base).val < 64) ∧ inActiveRing64Window t_server t_base ∧
    (∀ t_hist : Tick32, inActiveRing64Window t_server t_hist → t_hist ≠ t_base →
      ring64Slot t_hist ≠ ring64Slot t_base) := by
  dsimp only
  have h_in : inActiveRing64Window t_server (ackBaselineTick t_head b) := by
    unfold inActiveRing64Window ackBaselineTick; unfold tickDelta32 at hage ⊢
    rw [show t_server - (t_head - (b : Tick32)) = (t_server - t_head) + (b : Tick32) by abel,
        ZMod.val_add, ZMod.val_natCast, Nat.mod_eq_of_lt (by omega : b < 2 ^ 32)]; omega
  exact ⟨⟨hbit, (ring64Slot _).isLt⟩, h_in, fun t hh hne => (ring64_window_injective t_server t _ hh h_in).2 hne⟩

/-! ### Section 2: Causal Occluder Rewind Cutoff & Lag-Switch Non-Amplification -/
def morseJitterImpulse (j : ℤ → ℝ) (t : ℤ) : ℝ := (j t - j (t - 1)) * (j t - j (t + 1))
def wEff (κ rtt_half i_morse w_max : ℝ) : ℝ := min w_max (max 0 (rtt_half - κ * max 0 i_morse))

theorem morse_jitter_spike_characterization (j : ℤ → ℝ) (t : ℤ) :
    (0 < morseJitterImpulse j t ↔
      ((j (t - 1) < j t ∧ j (t + 1) < j t) ∨ (j t < j (t - 1) ∧ j t < j (t + 1)))) ∧
    (∀ j_base δ_burst : ℝ, δ_burst ≠ 0 → j (t - 1) = j_base → j (t + 1) = j_base →
      j t = j_base + δ_burst → morseJitterImpulse j t = δ_burst ^ 2 ∧ 0 < morseJitterImpulse j t) ∧
    (∀ j₀ α : ℝ, (∀ k : ℤ, j k = j₀ + α * (k : ℝ)) →
      morseJitterImpulse j t = -(α ^ 2) ∧ morseJitterImpulse j t ≤ 0 ∧ max 0 (morseJitterImpulse j t) = 0) := by
  refine ⟨by unfold morseJitterImpulse; rw [mul_pos_iff]; simp only [sub_pos, sub_neg],
    fun j_base δ hδ hp hn hc => ?_, fun j₀ α hlin => ?_⟩
  · have hsq : morseJitterImpulse j t = δ ^ 2 := by unfold morseJitterImpulse; rw [hp, hn, hc]; ring
    exact ⟨hsq, by rw [hsq]; positivity⟩
  · have hv : morseJitterImpulse j t = -(α ^ 2) := by
      unfold morseJitterImpulse; rw [hlin t, hlin (t - 1), hlin (t + 1)]; push_cast; ring
    have hnp : morseJitterImpulse j t ≤ 0 := by rw [hv]; nlinarith [sq_nonneg α]
    exact ⟨hv, hnp, max_eq_left hnp⟩
theorem lag_switch_rewind_non_amplification (κ rtt_half i_morse w_max : ℝ) (hκ : 0 < κ) :
    (0 ≤ i_morse → wEff κ rtt_half i_morse w_max ≤ wEff κ rtt_half 0 w_max) ∧
    (∀ i₁ i₂ : ℝ, 0 ≤ i₁ → i₁ ≤ i₂ → wEff κ rtt_half i₂ w_max ≤ wEff κ rtt_half i₁ w_max) ∧
    (0 < i_morse → 0 < rtt_half → rtt_half ≤ w_max → wEff κ rtt_half i_morse w_max < wEff κ rtt_half 0 w_max) := by
  have hmono : ∀ i₁ i₂ : ℝ, 0 ≤ i₁ → i₁ ≤ i₂ → wEff κ rtt_half i₂ w_max ≤ wEff κ rtt_half i₁ w_max :=
    fun i₁ i₂ h₁ h₁₂ => by
      unfold wEff; rw [max_eq_right h₁, max_eq_right (le_trans h₁ h₁₂)]
      exact min_le_min_left _ (max_le_max_left _ (by nlinarith))
  refine ⟨fun hi => hmono 0 i_morse le_rfl hi, hmono, fun hi hrtt hle => ?_⟩
  unfold wEff
  rw [max_eq_right (le_of_lt hi), max_self, mul_zero, sub_zero, max_eq_right (le_of_lt hrtt), min_eq_right hle]
  exact lt_of_le_of_lt (min_le_right _ _) (max_lt hrtt (by nlinarith))

def dot3 (u v : Fin 3 → ℝ) : ℝ := u 0 * v 0 + u 1 * v 1 + u 2 * v 2
def normSq3 (v : Fin 3 → ℝ) : ℝ := dot3 v v
def IsConvexShadow3 (Shadow : Set (Fin 3 → ℝ)) : Prop :=
  ∀ x ∈ Shadow, ∀ y ∈ Shadow, ∀ α : ℝ, 0 ≤ α → α ≤ 1 → (fun k => (1 - α) * x k + α * y k) ∈ Shadow
def inClampedRewindInterval (t_rewind t_server tau_fair t : ℝ) : Prop :=
  max t_rewind (t_server - tau_fair) ≤ t ∧ t ≤ t_server
def hitAdmissible (Shadow : Set (Fin 3 → ℝ)) (pos : ℝ → (Fin 3 → ℝ))
    (t_rewind t_server tau_fair t_shot : ℝ) : Prop :=
  inClampedRewindInterval t_rewind t_server tau_fair t_shot ∧ pos t_shot ∉ Shadow
theorem mutual_occluder_cutoff_eliminates_behind_wall_hits
    (Shadow : Set (Fin 3 → ℝ)) (pos : ℝ → (Fin 3 → ℝ)) (v_max t_occ t_rewind t_server tau_fair : ℝ)
    (hv_max : 0 ≤ v_max) (htau_nonneg : 0 ≤ tau_fair) (h_convex : IsConvexShadow3 Shadow)
    (h_speed : ∀ t₁ t₂ : ℝ, normSq3 (fun k => pos t₁ k - pos t₂ k) ≤ (v_max * (t₁ - t₂)) ^ 2)
    (h_occluded : ∀ t : ℝ, t_occ < t → t ≤ t_server → pos t ∈ Shadow) (h_past_fair : tau_fair < t_server - t_occ) :
    (t_rewind ≤ t_server → inClampedRewindInterval t_rewind t_server tau_fair t_server) ∧
    (∀ t : ℝ, inClampedRewindInterval t_rewind t_server tau_fair t →
      (t_occ < t ∧ t ≤ t_server) ∧ pos t ∈ Shadow ∧
      normSq3 (fun k => pos t k - pos t_server k) ≤ (v_max * tau_fair) ^ 2) ∧
    (∀ t₁ t₂ α : ℝ, inClampedRewindInterval t_rewind t_server tau_fair t₁ →
      inClampedRewindInterval t_rewind t_server tau_fair t₂ → 0 ≤ α → α ≤ 1 →
      (fun k => (1 - α) * pos t₁ k + α * pos t₂ k) ∈ Shadow) ∧
    (∀ t_shot : ℝ, hitAdmissible Shadow pos t_rewind t_server tau_fair t_shot ↔ False) := by
  have h_win : ∀ t : ℝ, inClampedRewindInterval t_rewind t_server tau_fair t →
      (t_occ < t ∧ t ≤ t_server) ∧ pos t ∈ Shadow ∧
      normSq3 (fun k => pos t k - pos t_server k) ≤ (v_max * tau_fair) ^ 2 := by
    rintro t ⟨ht_lo, ht_hi⟩
    have hf : t_server - tau_fair ≤ t := le_trans (le_max_right _ _) ht_lo
    have hgt : t_occ < t := by linarith
    have hsp : normSq3 (fun k => pos t k - pos t_server k) ≤ (v_max * tau_fair) ^ 2 := by
      nlinarith [h_speed t t_server, mul_le_mul_of_nonneg_left (show t_server - t ≤ tau_fair by linarith) hv_max,
        mul_nonneg hv_max (show 0 ≤ t_server - t by linarith)]
    exact ⟨⟨hgt, ht_hi⟩, h_occluded t hgt ht_hi, hsp⟩
  exact ⟨fun hr => ⟨max_le hr (by linarith), le_rfl⟩, h_win,
    fun t₁ t₂ α h₁ h₂ hα0 hα1 => h_convex _ (h_win t₁ h₁).2.1 _ (h_win t₂ h₂).2.1 α hα0 hα1,
    fun t_shot => ⟨fun ⟨hin, hnot⟩ => hnot (h_win t_shot hin).2.1, False.elim⟩⟩

/-! ### Section 3: Two-Sided Minkowski Reachability PVS & Anti-DMA Non-Interference -/
def ReachBall {Pos : Type*} [PseudoMetricSpace Pos] (x_j : Pos) (v_max dt : ℝ) : Set Pos :=
  { x | dist x x_j ≤ v_max * dt }
def causalPredeliveryWindow (rtt_half tau_margin : ℝ) : ℝ := rtt_half + tau_margin
def visMinkowski {Pos : Type*} [PseudoMetricSpace Pos] (F : Set Pos) (v_max dt : ℝ) (x_j : Pos) : Prop :=
  ∃ x ∈ F, dist x x_j ≤ v_max * dt
theorem minkowski_pvs_zero_pop_in {Pos : Type*} [PseudoMetricSpace Pos]
    (Frustum_i : Set Pos) (x_j : ℝ → Pos) (v_max rtt_half tau_margin t0 t : ℝ)
    (hv_max : 0 ≤ v_max) (hdt_nonneg : 0 ≤ causalPredeliveryWindow rtt_half tau_margin)
    (h_traj : ∀ τ : ℝ, t0 ≤ τ → τ ≤ t0 + causalPredeliveryWindow rtt_half tau_margin →
      dist (x_j τ) (x_j t0) ≤ v_max * (τ - t0))
    (ht_lo : t0 ≤ t) (ht_hi : t ≤ t0 + causalPredeliveryWindow rtt_half tau_margin) (h_peek : x_j t ∈ Frustum_i) :
    let dt := causalPredeliveryWindow rtt_half tau_margin
    x_j t ∈ ReachBall (x_j t0) v_max dt ∧ visMinkowski Frustum_i v_max dt (x_j t0) := by
  dsimp only; have _ := hdt_nonneg
  have hrad := le_trans (h_traj t ht_lo ht_hi) (mul_le_mul_of_nonneg_left
    (show t - t0 ≤ causalPredeliveryWindow rtt_half tau_margin by linarith) hv_max)
  exact ⟨hrad, ⟨x_j t, h_peek, hrad⟩⟩
theorem two_sided_minkowski_pvs_zero_pop_in {Pos : Type*} [PseudoMetricSpace Pos]
    (Frustum_i_t0 : Set Pos) (x_j : ℝ → Pos) (v_obs v_enemy dt t0 t : ℝ)
    (hv_obs : 0 ≤ v_obs) (hv_enemy : 0 ≤ v_enemy)
    (h_traj : ∀ τ : ℝ, t0 ≤ τ → τ ≤ t0 + dt → dist (x_j τ) (x_j t0) ≤ v_enemy * (τ - t0))
    (ht_lo : t0 ≤ t) (ht_hi : t ≤ t0 + dt)
    (h_two_sided_peek : ∃ z ∈ Frustum_i_t0, dist (x_j t) z ≤ v_obs * (t - t0)) :
    visMinkowski Frustum_i_t0 (v_obs + v_enemy) dt (x_j t0) := by
  rcases h_two_sided_peek with ⟨z, hz_in, hz_dist⟩
  have htri := dist_triangle z (x_j t) (x_j t0); rw [dist_comm z (x_j t)] at htri
  exact ⟨z, hz_in, le_trans (show dist z (x_j t0) ≤ (v_obs + v_enemy) * (t - t0) by
    linarith [htri, hz_dist, h_traj t ht_lo ht_hi])
    (mul_le_mul_of_nonneg_left (by linarith) (add_nonneg hv_obs hv_enemy))⟩

structure RaidActorState (Pos Loot : Type*) where
  pos : Pos
  armorPlates : UInt32
  health : UInt16
  loot : Loot
  deriving DecidableEq
structure RaidState (N : ℕ) (Pos Loot Byte : Type*) where
  actors : Fin N → RaidActorState Pos Loot
  anonymousAcoustics : List Byte
noncomputable def packDatagram {N : ℕ} {Pos Loot Byte : Type*} [PseudoMetricSpace Pos]
    (Frustum : Fin N → Set Pos) (v_max dt : ℝ)
    (encodeActor : Fin N → Fin N → RaidActorState Pos Loot → List Byte)
    (i : Fin N) (state : RaidState N Pos Loot Byte) : List Byte :=
  open Classical in ((List.finRange N).filter (fun j =>
    decide (visMinkowski (Frustum i) v_max dt (state.actors j).pos))).flatMap
      (fun j => encodeActor i j (state.actors j)) ++ state.anonymousAcoustics
theorem anti_dma_observational_noninterference {N : ℕ} {Pos Loot Byte : Type*}
    [PseudoMetricSpace Pos] (Frustum : Fin N → Set Pos) (v_max dt : ℝ)
    (encodeActor : Fin N → Fin N → RaidActorState Pos Loot → List Byte)
    (i : Fin N) (s₁ s₂ : RaidState N Pos Loot Byte)
    (h_vis_eq : { j : Fin N | visMinkowski (Frustum i) v_max dt (s₁.actors j).pos } =
                { j : Fin N | visMinkowski (Frustum i) v_max dt (s₂.actors j).pos })
    (h_vis_agree : ∀ j : Fin N, visMinkowski (Frustum i) v_max dt (s₁.actors j).pos → s₁.actors j = s₂.actors j)
    (h_acoustics : s₁.anonymousAcoustics = s₂.anonymousAcoustics) :
    packDatagram Frustum v_max dt encodeActor i s₁ = packDatagram Frustum v_max dt encodeActor i s₂ := by
  classical
  unfold packDatagram
  have hpred : (fun j : Fin N => decide (visMinkowski (Frustum i) v_max dt (s₁.actors j).pos)) =
      (fun j : Fin N => decide (visMinkowski (Frustum i) v_max dt (s₂.actors j).pos)) := by
    funext j
    have hj : visMinkowski (Frustum i) v_max dt (s₁.actors j).pos ↔
      visMinkowski (Frustum i) v_max dt (s₂.actors j).pos := Set.ext_iff.mp h_vis_eq j
    simp [hj]
  rw [← congr_arg (List.filter · (List.finRange N)) hpred, h_acoustics]
  exact congr_arg (· ++ s₂.anonymousAcoustics) (List.flatMap_congr fun j hj => by
    rw [h_vis_agree j (of_decide_eq_true (List.mem_filter.mp hj).2)])
theorem anti_dma_occluded_mutation_zero_leak {N : ℕ} {Pos Loot Byte : Type*}
    [PseudoMetricSpace Pos] (Frustum : Fin N → Set Pos) (v_max dt : ℝ)
    (encodeActor : Fin N → Fin N → RaidActorState Pos Loot → List Byte)
    (i j_occ : Fin N) (s : RaidState N Pos Loot Byte)
    (pos' : Pos) (armor' : UInt32) (hp' : UInt16) (loot' : Loot)
    (h_occ_before : ¬ visMinkowski (Frustum i) v_max dt (s.actors j_occ).pos)
    (h_occ_after : ¬ visMinkowski (Frustum i) v_max dt pos') :
    let s_mut : RaidState N Pos Loot Byte :=
      { actors := fun j => if j = j_occ then ⟨pos', armor', hp', loot'⟩ else s.actors j
        anonymousAcoustics := s.anonymousAcoustics }
    packDatagram Frustum v_max dt encodeActor i s_mut = packDatagram Frustum v_max dt encodeActor i s := by
  refine anti_dma_observational_noninterference Frustum v_max dt encodeActor i _ s ?_ ?_ rfl
  · ext j; rcases eq_or_ne j j_occ with rfl | hj
    · simp [h_occ_before, h_occ_after]
    · simp [hj]
  · intro j hv; rcases eq_or_ne j j_occ with rfl | hj
    · simp [h_occ_after] at hv
    · simp [hj]

/-! ### Section 4: 3D Cayley Attitude Rotor & Swept-Capsule Collide-and-Slide -/
def eye3 (i j : Fin 3) : ℝ := if i = j then 1 else 0
def skewSO3 (ω : Fin 3 → ℝ) (i j : Fin 3) : ℝ :=
  match i.val, j.val with
  | 0, 1 => -ω 2 | 0, 2 => ω 1 | 1, 0 => ω 2
  | 1, 2 => -ω 0 | 2, 0 => -ω 1 | 2, 1 => ω 0 | _, _ => 0
def cayleyDenomSO3 (ω : Fin 3 → ℝ) : ℝ := 1 + ω 0 ^ 2 + ω 1 ^ 2 + ω 2 ^ 2
def cayleyNumSO3 (ω : Fin 3 → ℝ) (i j : Fin 3) : ℝ :=
  let x := ω 0; let y := ω 1; let z := ω 2
  match i.val, j.val with
  | 0, 0 => 1 + x ^ 2 - y ^ 2 - z ^ 2 | 0, 1 => 2 * x * y - 2 * z | 0, 2 => 2 * x * z + 2 * y
  | 1, 0 => 2 * x * y + 2 * z | 1, 1 => 1 - x ^ 2 + y ^ 2 - z ^ 2 | 1, 2 => 2 * y * z - 2 * x
  | 2, 0 => 2 * x * z - 2 * y | 2, 1 => 2 * y * z + 2 * x | 2, 2 => 1 - x ^ 2 - y ^ 2 + z ^ 2
  | _, _ => 0
noncomputable def cayleyRotorSO3 (ω : Fin 3 → ℝ) (i j : Fin 3) : ℝ := cayleyNumSO3 ω i j / cayleyDenomSO3 ω
def matVec3 (M : Fin 3 → Fin 3 → ℝ) (v : Fin 3 → ℝ) (i : Fin 3) : ℝ := ∑ j : Fin 3, M i j * v j
def cross3 (u v : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  match k.val with
  | 0 => u 1 * v 2 - u 2 * v 1 | 1 => u 2 * v 0 - u 0 * v 2 | _ => u 0 * v 1 - u 1 * v 0
lemma sum_fin3 (f : Fin 3 → ℝ) : (∑ i : Fin 3, f i) = f 0 + f 1 + f 2 := by
  rw [show (univ : Finset (Fin 3)) = {0, 1, 2} by decide]; simp [Finset.sum_insert]; ring
theorem cayley_so3_attitude_orthogonality_and_norm_conservation (ω v : Fin 3 → ℝ) :
    (∀ i j : Fin 3, skewSO3 ω j i = -skewSO3 ω i j) ∧
    (∀ i k : Fin 3, (∑ j : Fin 3, (eye3 i j - skewSO3 ω i j) * cayleyRotorSO3 ω j k) = eye3 i k + skewSO3 ω i k) ∧
    (∀ j k : Fin 3, (∑ i : Fin 3, cayleyRotorSO3 ω i j * cayleyRotorSO3 ω i k) = eye3 j k) ∧
    normSq3 (matVec3 (cayleyRotorSO3 ω) v) = normSq3 v := by
  have hΔ : cayleyDenomSO3 ω ≠ 0 := ne_of_gt (by unfold cayleyDenomSO3; positivity)
  have h_res : ∀ i k : Fin 3, (∑ j : Fin 3, (eye3 i j - skewSO3 ω i j) * cayleyRotorSO3 ω j k) =
      eye3 i k + skewSO3 ω i k := fun i k => by
    unfold cayleyRotorSO3; rw [sum_fin3]; simp only [mul_div_assoc', ← add_div]
    rw [show ((eye3 i 0 - skewSO3 ω i 0) * cayleyNumSO3 ω 0 k + (eye3 i 1 - skewSO3 ω i 1) * cayleyNumSO3 ω 1 k +
        (eye3 i 2 - skewSO3 ω i 2) * cayleyNumSO3 ω 2 k) = (eye3 i k + skewSO3 ω i k) * cayleyDenomSO3 ω by
      fin_cases i <;> fin_cases k <;> simp [eye3, skewSO3, cayleyNumSO3, cayleyDenomSO3] <;> ring,
      mul_div_cancel_right₀ _ hΔ]
  have h_orth : ∀ j k : Fin 3, (∑ i : Fin 3, cayleyRotorSO3 ω i j * cayleyRotorSO3 ω i k) = eye3 j k := fun j k => by
    unfold cayleyRotorSO3; rw [sum_fin3]; simp only [div_mul_div_comm, ← sq, ← add_div]
    rw [show cayleyNumSO3 ω 0 j * cayleyNumSO3 ω 0 k + cayleyNumSO3 ω 1 j * cayleyNumSO3 ω 1 k +
        cayleyNumSO3 ω 2 j * cayleyNumSO3 ω 2 k = eye3 j k * cayleyDenomSO3 ω ^ 2 by
      fin_cases j <;> fin_cases k <;> simp [eye3, cayleyNumSO3, cayleyDenomSO3] <;> ring,
      mul_div_cancel_right₀ _ (pow_ne_zero 2 hΔ)]
  refine ⟨fun i j => by fin_cases i <;> fin_cases j <;> simp [skewSO3], h_res, h_orth, ?_⟩
  have h00 := h_orth 0 0; have h01 := h_orth 0 1; have h02 := h_orth 0 2
  have h11 := h_orth 1 1; have h12 := h_orth 1 2; have h22 := h_orth 2 2
  rw [sum_fin3] at h00 h01 h02 h11 h12 h22
  simp only [eye3, Fin.isValue, Fin.reduceEq, ↓reduceIte] at h00 h01 h02 h11 h12 h22
  unfold normSq3 dot3 matVec3; rw [sum_fin3, sum_fin3, sum_fin3]
  linear_combination (v 0 ^ 2) * h00 + (2 * v 0 * v 1) * h01 + (2 * v 0 * v 2) * h02 +
    (v 1 ^ 2) * h11 + (2 * v 1 * v 2) * h12 + (v 2 ^ 2) * h22

lemma normSq3_nonneg (u : Fin 3 → ℝ) : 0 ≤ normSq3 u :=
  add_nonneg (add_nonneg (mul_self_nonneg (u 0)) (mul_self_nonneg (u 1))) (mul_self_nonneg (u 2))
lemma normSq3_sub_scale (v n : Fin 3 → ℝ) (m : ℝ) :
    normSq3 (fun k => v k - m * n k) - normSq3 v = m ^ 2 * normSq3 n - 2 * (m * dot3 v n) := by
  unfold normSq3 dot3; dsimp only; ring
lemma normSq3_orth_decomp (v w : Fin 3 → ℝ) (horth : dot3 (fun k => v k - w k) w = 0) :
    normSq3 v = normSq3 w + normSq3 (fun k => v k - w k) := by
  unfold normSq3; unfold dot3 at horth ⊢; dsimp only at horth ⊢; linear_combination 2 * horth
def collideAndSlideSingle (v n : Fin 3 → ℝ) (k : Fin 3) : ℝ := v k - min 0 (dot3 v n) * n k
noncomputable def creaseSlideVelocity (v n₁ n₂ : Fin 3 → ℝ) (k : Fin 3) : ℝ :=
  (dot3 v (cross3 n₁ n₂) / normSq3 (cross3 n₁ n₂)) * cross3 n₁ n₂ k
theorem swept_capsule_collide_and_slide_non_penetration (v n : Fin 3 → ℝ) (h_unit : normSq3 n = 1) :
    let v' := collideAndSlideSingle v n
    0 ≤ dot3 v' n ∧ dot3 v' n = max 0 (dot3 v n) ∧
    (∀ t : Fin 3 → ℝ, dot3 t n = 0 → dot3 v' t = dot3 v t) ∧ normSq3 v' ≤ normSq3 v := by
  dsimp only
  have hn_dot : n 0 * n 0 + n 1 * n 1 + n 2 * n 2 = 1 := h_unit
  have hdot_n : dot3 (collideAndSlideSingle v n) n = dot3 v n - min 0 (dot3 v n) := by
    unfold collideAndSlideSingle dot3; dsimp only
    linear_combination -min 0 (v 0 * n 0 + v 1 * n 1 + v 2 * n 2) * hn_dot
  have h_tan : ∀ t : Fin 3 → ℝ, dot3 t n = 0 → dot3 (collideAndSlideSingle v n) t = dot3 v t := by
    intro t htn; have htn_s : n 0 * t 0 + n 1 * t 1 + n 2 * t 2 = 0 := by unfold dot3 at htn; linarith
    unfold collideAndSlideSingle dot3; dsimp only
    linear_combination -min 0 (v 0 * n 0 + v 1 * n 1 + v 2 * n 2) * htn_s
  have hexp := normSq3_sub_scale v n (min 0 (dot3 v n)); rw [h_unit, mul_one] at hexp
  change normSq3 (collideAndSlideSingle v n) - normSq3 v = _ at hexp
  rcases le_total 0 (dot3 v n) with h | h
  · rw [min_eq_left h] at hdot_n hexp; rw [hdot_n, max_eq_right h, sub_zero]; exact ⟨h, rfl, h_tan, by linarith⟩
  · rw [min_eq_right h] at hdot_n hexp; rw [hdot_n, max_eq_left h, sub_self]
    exact ⟨le_rfl, rfl, h_tan, by nlinarith [sq_nonneg (dot3 v n)]⟩
theorem two_plane_crease_projection_orthogonal (v n₁ n₂ : Fin 3 → ℝ) :
    let v_crease := creaseSlideVelocity v n₁ n₂
    (dot3 v_crease n₁ = 0 ∧ dot3 v_crease n₂ = 0) ∧
    (0 < normSq3 (cross3 n₁ n₂) →
      dot3 v_crease (cross3 n₁ n₂) = dot3 v (cross3 n₁ n₂) ∧ normSq3 v_crease ≤ normSq3 v) := by
  dsimp only; set c := cross3 n₁ n₂ with hc_def; set s := dot3 v c / normSq3 c with hs_def
  have hvc : creaseSlideVelocity v n₁ n₂ = fun k => s * c k := rfl
  have hsc : ∀ u : Fin 3 → ℝ, dot3 (fun k => s * c k) u = s * dot3 c u := fun u => by unfold dot3; ring
  have hc1 : dot3 c n₁ = 0 := by rw [hc_def]; unfold dot3 cross3; simp; ring
  have hc2 : dot3 c n₂ = 0 := by rw [hc_def]; unfold dot3 cross3; simp; ring
  refine ⟨⟨by rw [hvc, hsc, hc1, mul_zero], by rw [hvc, hsc, hc2, mul_zero]⟩, fun hc_pos => ?_⟩
  have h_along : dot3 (creaseSlideVelocity v n₁ n₂) c = dot3 v c := by
    rw [hvc, hsc]; change s * normSq3 c = dot3 v c; rw [hs_def, div_mul_cancel₀ _ (ne_of_gt hc_pos)]
  have horth : dot3 (fun k => v k - creaseSlideVelocity v n₁ n₂ k) (creaseSlideVelocity v n₁ n₂) = 0 := by
    have hsymm : ∀ a b : Fin 3 → ℝ, dot3 a b = dot3 b a := fun a b => by unfold dot3; ring
    have hsub : dot3 (fun k => v k - creaseSlideVelocity v n₁ n₂ k) c = 0 := by
      rw [show dot3 (fun k => v k - creaseSlideVelocity v n₁ n₂ k) c =
        dot3 v c - dot3 (creaseSlideVelocity v n₁ n₂) c by unfold dot3; ring, h_along, sub_self]
    rw [hsymm]; conv_lhs => arg 1; rw [hvc]
    rw [hsc, hsymm c, hsub, mul_zero]
  have hp := normSq3_orth_decomp v (creaseSlideVelocity v n₁ n₂) horth
  have hn := normSq3_nonneg (fun k => v k - creaseSlideVelocity v n₁ n₂ k)
  exact ⟨h_along, by linarith⟩

/-! ### Section 5: `NetTwinArena` `1.50 MB` Layout, Disjointness & `320 KB` L2 Bound -/
def netTwinOffsets : Fin 8 → ℕ
  | 0 => 0 | 1 => 65536 | 2 => 327680 | 3 => 589824
  | 4 => 655360 | 5 => 720896 | 6 => 1245184 | 7 => 1507328
def netTwinSizes : Fin 8 → ℕ
  | 0 => 65536 | 1 => 262144 | 2 => 262144 | 3 => 65536
  | 4 => 65536 | 5 => 524288 | 6 => 262144 | 7 => 65536
def netTwinTotalBytes : ℕ := 1572864
def cortexA76PrivateL2Bytes : ℕ := 524288
def bcm2712SharedL3Bytes : ℕ := 2097152
def clientInputCmdBytes : ℕ := 4 + 2 + 4 + 4 + 1 + 1
def hitRecordOutBytes : ℕ := 4 + 4 + 4 + 2 + 1 + 1 + 4 + 4 + 4 + 4
def acousticImpulseBytes : ℕ := 4 + 2 + 2 + 4
def playerPoseSnapshotBytes : ℕ := 4 * 6 + 2 * 4
theorem net_twin_arena_layout_verified :
    (∀ i : Fin 8, netTwinOffsets i % 64 = 0) ∧
    (∀ i j : Fin 8, i.val < j.val → netTwinOffsets i + netTwinSizes i ≤ netTwinOffsets j) ∧
    (netTwinSizes 0 + netTwinSizes 1 = 327680 ∧ netTwinSizes 0 + netTwinSizes 1 ≤ cortexA76PrivateL2Bytes) ∧
    (netTwinOffsets 7 + netTwinSizes 7 = netTwinTotalBytes ∧ (∑ i : Fin 8, netTwinSizes i) = netTwinTotalBytes ∧
      netTwinTotalBytes = 1572864 ∧ netTwinTotalBytes < bcm2712SharedL3Bytes) ∧
    (clientInputCmdBytes = 16 ∧ hitRecordOutBytes = 32 ∧
      acousticImpulseBytes = 12 ∧ playerPoseSnapshotBytes = 32) := by decide

/-! ### Section 6: v0.2 Modular Baseline-Delta Reconstruction & 30-Plate Monotonicity -/
def encodeDeltaChain {α : Type} [Sub α] : α → List α → List α
  | _, [] => []
  | prev, cur :: rest => (cur - prev) :: encodeDeltaChain cur rest
def decodeDeltaChain {α : Type} [Add α] (base : α) (deltas : List α) : α := deltas.foldl (· + ·) base
lemma decode_encode_chain {α : Type} [Add α] [Sub α]
    (hcancel : ∀ a b : α, a + (b - a) = b) (base cur : α) (hops : List α) :
    decodeDeltaChain base (encodeDeltaChain base (hops ++ [cur])) = cur := by
  have hfold : ∀ (b : α) (xs : List α),
      decodeDeltaChain b (encodeDeltaChain b xs) = xs.foldl (fun _ x => x) b := by
    intro b xs; induction xs generalizing b with
    | nil => rfl
    | cons x rest ih => simpa [encodeDeltaChain, decodeDeltaChain, hcancel b x] using ih x
  have hlast : ∀ (b : α) (xs : List α), (xs ++ [cur]).foldl (fun _ y => y) b = cur := by
    intro b xs; induction xs generalizing b with
    | nil => rfl
    | cons y rest ih => exact ih y
  exact (hfold base (hops ++ [cur])).trans (hlast base hops)
theorem modular_delta_chain_exact_reconstruction (M : ℕ)
    (base cur : ZMod M) (hops : List (ZMod M))
    (b8 c8 : UInt8) (hops8 : List UInt8) (b16 c16 : UInt16) (hops16 : List UInt16) :
    base + (cur - base) = cur ∧ decodeDeltaChain base (encodeDeltaChain base (hops ++ [cur])) = cur ∧
    b8 + (c8 - b8) = c8 ∧ decodeDeltaChain b8 (encodeDeltaChain b8 (hops8 ++ [c8])) = c8 ∧
    b16 + (c16 - b16) = c16 ∧ decodeDeltaChain b16 (encodeDeltaChain b16 (hops16 ++ [c16])) = c16 :=
  have hc8 : ∀ a b : UInt8, a + (b - a) = b := fun a b => by rw [UInt8.add_comm, UInt8.sub_add_cancel]
  have hc16 : ∀ a b : UInt16, a + (b - a) = b := fun a b => by rw [UInt16.add_comm, UInt16.sub_add_cancel]
  ⟨add_sub_cancel base cur, decode_encode_chain add_sub_cancel base cur hops,
    hc8 b8 c8, decode_encode_chain hc8 b8 c8 hops8,
    hc16 b16 c16, decode_encode_chain hc16 b16 c16 hops16⟩

def stepPlateDamage (plates : Fin 30 → ℕ) (zone : Fin 30) (dmg : ℕ) : Fin 30 → ℕ :=
  Function.update plates zone (plates zone - dmg)
def applyPlateImpacts (plates : Fin 30 → ℕ) (impacts : List (Fin 30 × ℕ)) : Fin 30 → ℕ :=
  impacts.foldl (fun p hit => stepPlateDamage p hit.1 hit.2) plates
theorem multi_plate_durability_monotone_bounded
    (plates : Fin 30 → ℕ) (zone : Fin 30) (dmg : ℕ) (impacts : List (Fin 30 × ℕ)) :
    (∀ z : Fin 30, stepPlateDamage plates zone dmg z ≤ plates z) ∧
    (∀ z : Fin 30, z ≠ zone → stepPlateDamage plates zone dmg z = plates z) ∧
    (∀ z : Fin 30, applyPlateImpacts plates impacts z ≤ plates z) ∧
    ((∀ z : Fin 30, plates z ≤ 255) → ∀ z : Fin 30, applyPlateImpacts plates impacts z ≤ 255) := by
  have hstep : ∀ (p : Fin 30 → ℕ) (zn : Fin 30) (d : ℕ) (z : Fin 30),
      stepPlateDamage p zn d z ≤ p z := fun p zn d z => by
    unfold stepPlateDamage Function.update; split_ifs with h
    · exact h ▸ Nat.sub_le _ _
    · exact le_rfl
  have hother : ∀ z : Fin 30, z ≠ zone → stepPlateDamage plates zone dmg z = plates z := fun z hne => by
    unfold stepPlateDamage Function.update; split_ifs with h
    · exact absurd h hne
    · rfl
  have hfold : ∀ (p : Fin 30 → ℕ) (xs : List (Fin 30 × ℕ)) (z : Fin 30),
      applyPlateImpacts p xs z ≤ p z := by
    intro p xs; induction xs generalizing p with
    | nil => exact fun _ => le_rfl
    | cons hit rest ih => exact fun z => le_trans (ih _ z) (hstep p hit.1 hit.2 z)
  exact ⟨hstep plates zone dmg, hother, hfold plates impacts,
    fun h255 z => le_trans (hfold plates impacts z) (h255 z)⟩
end NetSnail
