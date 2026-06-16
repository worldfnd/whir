//! Zook prover — Construction 9.7 (ZK Code-Switching).
//!
//! Per ZK round, the orchestrator wraps three sub-protocols:
//!   - **ZK sumcheck** (Lemma 6.5): reduces `<message, covector> = sum`,
//!     consuming `k` short (degree-2) mask polynomials for hiding.
//!   - **Code-switch** (Construction 9.7): reduces source IRS to a target
//!     IRS, consuming one `cs_mask = (r ‖ s)` mask oracle of length ℓ_zk.
//!   - **Mask proximity** (Construction 7.2): proves the committed masks are
//!     close to C_zk codewords.
//!
//! The mask oracle is split across **two** C_zk trees:
//!   - `sumcheck_masks` tree: k masks at vector_size = next_pow_2(mask_length)
//!     (= 4 for degree-2 round polys). Committed BEFORE sumcheck — standard
//!     ZK-sumcheck discipline.
//!   - `cs_mask` tree: 1 mask at vector_size = ℓ_zk. Committed AFTER sumcheck
//!     so cs_mask's `r` part can carry `fold(source.masks, folding_randomness)`
//!     (Construction 9.7 `(r ‖ s)` structure).
//!
//! Splitting avoids padding the tiny sumcheck masks up to ℓ_zk; the NTT and
//! Merkle work at C_zk's rate drops by ~4× in typical configurations.
//!
//! Per-round flow:
//!   1. Sample sumcheck masks `M_0..M_{k−1}` (length `mask_length` each) and
//!      `cs_fresh_padding` of length `l_zk − source.mask_length()`.
//!   2. Commit `sumcheck_masks` tree (k masks padded to next_pow_2(mask_length)).
//!   3. Run ZK sumcheck. Post-sumcheck `state.sum = δ + γ_sumcheck · dot`,
//!      where `δ = Σ M_i(r_i)`.
//!   4. Derive `folded_irs_masks = fold(source.masks, r_0..r_{k−1})` (Identity<F>:
//!      no lift needed), assemble `cs_mask = (folded_irs_masks ‖ cs_fresh_padding)`,
//!      commit `cs_mask` tree.
//!   5. Send `δ` cleartext and reconcile
//!      `state.sum := (sum − δ) · γ_sumcheck⁻¹` so the claim entering
//!      code-switch is the unmasked `dot(folded_a, post_sumcheck_cov)`.
//!   6. Code-switch updates `state.{message, irs_witness, covector, sum}`.
//!   7. Prove sumcheck masks at `[1, r_i, …, r_i^{sumcheck_vec_size−1}]`
//!      (gives X_i = M_i(r_i)); prove cs_mask at the post-cs covector mask
//!      region (gives X_cs). The verifier checks `Σ X_{i<k} == δ` (binds δ);
//!      both sides subtract X_cs to project sum to f-only.
//!
//! After all rounds: commit the final folded message via `basecase.commit`
//! and run `basecase.prove`. Basecase-only plans skip the loop.

use ark_ff::Field;
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, RngCore};
#[cfg(feature = "tracing")]
use tracing::instrument;
use zeroize::Zeroize;

use crate::{
    algebra::{
        dot, embedding::Identity, geometric_sequence, linear_form::LinearForm, random_vector,
        univariate_evaluate,
    },
    hash::Hash,
    protocols::{
        code_switch::{self, fold_chunks},
        irs_commit::Witness as IrsWitness,
        mask_proximity,
        mask_proximity::Config as MaskProximityConfig,
        params::protocol_config::{MaskOracleConfig, ProtocolConfig, RoundConfig},
        sumcheck::{SumcheckMode, SumcheckOpening},
        zook::commit::{CommittedState, CommittedWitness},
    },
    transcript::{
        codecs::U64, Codec, Decoding, DuplexSpongeInterface, ProverMessage, ProverState,
        VerifierMessage,
    },
};

impl<F: Field + Default + Zeroize> ProtocolConfig<Identity<F>> {
    /// Prove `f(witness) == evaluations[j]` for every linear_form `f = linear_forms[j]` against
    /// the committed witness. Consumes `committed`.
    #[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::prove", fields(vector_size = self.tuning().vector_size, num_rounds = self.rounds().len(), num_claims = linear_forms.len())))]
    pub fn prove<H, R>(
        &self,
        ps: &mut ProverState<H, R>,
        committed: CommittedWitness<Identity<F>>,
        linear_forms: &[&dyn LinearForm<F>],
        evaluations: &[F],
    ) where
        Standard: Distribution<F>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        F: Codec<[H::U]>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        assert_eq!(
            linear_forms.len(),
            evaluations.len(),
            "linear_forms.len() != evaluations.len()"
        );
        assert!(
            !linear_forms.is_empty(),
            "zook requires ≥ 1 (form, value) pair"
        );

        // RLC challenge binds the form/value set to the commitment.
        let batching_challenge: F = ps.verifier_message();
        let claim_weights = geometric_sequence(batching_challenge, linear_forms.len());

        // Materialize the combined covector = Σ γ^j · form_j and combined value.
        let mut covector = vec![F::ZERO; self.tuning().vector_size];
        for (form, &weight) in linear_forms.iter().zip(&claim_weights) {
            form.accumulate(&mut covector, weight);
        }
        let batched_evaluation: F = evaluations
            .iter()
            .zip(&claim_weights)
            .map(|(v, weight)| *v * weight)
            .sum();

        // Reduce to basecase inputs `(message, witness, covector, sum)`. The
        // two arms differ only in how those are obtained.
        let (message, basecase_witness, covector, sum) = match committed.state {
            // Basecase-only plan: use the committed witness directly.
            CommittedState::Basecase {
                message,
                irs_witness,
            } => (message, irs_witness, covector, batched_evaluation),
            CommittedState::Round {
                message,
                irs_witness,
            } => {
                let mut state = ProverRoundState {
                    message,
                    irs_witness,
                    covector,
                    sum: batched_evaluation,
                };
                for round in self.rounds() {
                    state = prove_round(round, state, ps);
                }
                // After per-round reconciliation, state.sum is bound to
                // dot(state.message, state.covector) — no extra transcript
                // send needed entering basecase.
                (state.message, state.irs_witness, state.covector, state.sum)
            }
        };

        // Standard mode (BasecaseMode::Standard) sends the full witness vector
        // and IRS randomness cleartext. Only call with Mode::ZeroKnowledge if
        // end-to-end hiding is required.
        let _ = self
            .basecase()
            .prove(ps, message, &basecase_witness, covector, sum);
    }
}

/// Per-round transient state. `irs_witness` is `IrsWitness<F>` throughout
/// because we restrict to `Identity<F>` embeddings (M::Source = M::Target = F).
struct ProverRoundState<F: Field> {
    message: Vec<F>,
    irs_witness: IrsWitness<F>,
    covector: Vec<F>,
    sum: F,
}

#[cfg_attr(feature = "tracing", instrument(skip_all, name = "zook::prove_round", fields(msg_len = round.code_switch().source().message_length(), message_len = state.message.len())))]
fn prove_round<F, H, R>(
    round: &RoundConfig<Identity<F>>,
    mut state: ProverRoundState<F>,
    ps: &mut ProverState<H, R>,
) -> ProverRoundState<F>
where
    F: Field + Default + Zeroize + Codec<[H::U]>,
    Standard: Distribution<F>,
    H: DuplexSpongeInterface,
    R: RngCore + CryptoRng,
    u8: Decoding<[H::U]>,
    [u8; 32]: Decoding<[H::U]>,
    U64: Codec<[H::U]>,
    Hash: ProverMessage<[H::U]>,
{
    let msg_len = round.code_switch().source().message_length();

    debug_assert_eq!(
        dot(&state.message, &state.covector),
        state.sum,
        "prove_round entry: dot(message, covector) must equal sum"
    );

    // Samples and commits the sumcheck-masks tree (ZK) or is a no-op (Standard).
    // cs_fresh_padding is pre-sampled here because it does not depend on folding randomness.
    let mut masker = RoundMaskOracle::begin(round, ps);

    let opening = round.sumcheck().prove(
        ps,
        &mut state.message,
        &mut state.covector,
        &mut state.sum,
        masker.sumcheck_blinding(),
    );

    // Build cs_mask = (folded_irs_masks ‖ cs_fresh_padding), commit its tree,
    // send mask_eval_sum cleartext, reconcile sum to the unmasked dot.
    masker.bind_code_switch_mask(&state.irs_witness, &opening, &mut state.sum, ps);

    debug_assert_eq!(
        dot(&state.message, &state.covector),
        state.sum,
        "post-reconcile: dot(message, covector) must equal sum"
    );

    // Extend covector for ZK mask region; +0 in Standard mode.
    state
        .covector
        .resize(msg_len + masker.covector_extension(), F::ZERO);
    let cs_witness = round.code_switch().prove(
        ps,
        state.message,
        std::mem::take(&mut state.irs_witness),
        code_switch::Claim {
            covector: &mut state.covector,
            sum: &mut state.sum,
        },
        &opening.round_challenges,
        masker.code_switch_blinding(),
    );

    // Prove both mask trees; subtract cs_mask contribution to project sum to f-only.
    masker.finish(
        &opening.round_challenges,
        &state.covector[msg_len..],
        &mut state.sum,
        ps,
    );
    drop(opening);

    state.message = cs_witness.message;
    state.irs_witness = cs_witness.target_witness;
    state.covector.truncate(state.message.len());

    debug_assert_eq!(
        dot(&state.message, &state.covector),
        state.sum,
        "prove_round exit: dot(message, covector) must equal sum"
    );

    state
}

/// The committed mask tree for the sumcheck sub-protocol.
/// Holds k blinding polynomials sampled before sumcheck and opened after code-switch.
struct SumcheckMaskTree<'a, F: Field> {
    cfg: &'a MaskProximityConfig<F>,
    padded_masks: Vec<Vec<F>>,
    flat_coefficients: Vec<F>,
    tree_witness: mask_proximity::Witness<F>,
    padded_vec_size: usize,
}

impl<'a, F: Field + Zeroize> SumcheckMaskTree<'a, F> {
    fn sample_and_commit<H, R>(
        cfg: &'a MaskProximityConfig<F>,
        num_masks: usize,
        mask_poly_len: usize,
        ps: &mut ProverState<H, R>,
    ) -> Self
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        Hash: ProverMessage<[H::U]>,
    {
        let padded_vec_size = cfg.c_zk_commit().vector_size();
        let flat_coefficients: Vec<F> = random_vector(ps.rng(), num_masks * mask_poly_len);
        let padded_masks: Vec<Vec<F>> = (0..num_masks)
            .map(|i| {
                let mut padded = vec![F::ZERO; padded_vec_size];
                padded[..mask_poly_len].copy_from_slice(
                    &flat_coefficients[i * mask_poly_len..(i + 1) * mask_poly_len],
                );
                padded
            })
            .collect();
        let padded_mask_refs: Vec<&[F]> = padded_masks.iter().map(Vec::as_slice).collect();
        let tree_witness = cfg.commit(ps, &padded_mask_refs);
        Self {
            cfg,
            padded_masks,
            flat_coefficients,
            tree_witness,
            padded_vec_size,
        }
    }

    /// Input to sumcheck.prove(): the flat blinding coefficients.
    fn blinding(&self) -> &[F] {
        &self.flat_coefficients
    }

    /// Zeroize the flat blinding coefficients once sumcheck no longer needs them.
    fn wipe_blinding(&mut self) {
        self.flat_coefficients.zeroize();
    }

    /// δ = Σ_i evaluate(padded_masks[i], challenges[i])
    fn eval_sum(&self, challenges: &[F]) -> F {
        challenges
            .iter()
            .zip(self.padded_masks.iter())
            .map(|(&c, mask)| univariate_evaluate(mask, c))
            .sum()
    }

    /// Open the sumcheck-masks tree at the per-mask geometric covectors.
    /// Zeroizes padded_masks before returning.
    fn prove<H, R>(mut self, round_challenges: &[F], ps: &mut ProverState<H, R>)
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let evaluation_covectors: Vec<Vec<F>> = round_challenges
            .iter()
            .map(|&c| geometric_sequence(c, self.padded_vec_size))
            .collect();
        let covector_refs: Vec<&[F]> = evaluation_covectors.iter().map(Vec::as_slice).collect();
        let padded_mask_refs: Vec<&[F]> = self.padded_masks.iter().map(Vec::as_slice).collect();
        self.cfg.prove(
            ps,
            self.tree_witness,
            &padded_mask_refs,
            Some(&covector_refs),
        );
        for mask in &mut self.padded_masks {
            mask.zeroize();
        }
    }
}

/// The committed mask tree for the code-switch sub-protocol.
/// Holds the single (r_folded ‖ fresh_padding) polynomial, built after sumcheck.
struct CodeSwitchMask<'a, F: Field> {
    cfg: &'a MaskProximityConfig<F>,
    poly: Vec<F>,
    tree_witness: mask_proximity::Witness<F>,
}

impl<'a, F: Field + Zeroize> CodeSwitchMask<'a, F> {
    /// Assemble poly = (r_folded ‖ fresh_padding) and commit the cs-mask tree.
    fn build_and_commit<H, R>(
        r_folded: &[F],
        mut fresh_padding: Vec<F>,
        cfg: &'a MaskProximityConfig<F>,
        ps: &mut ProverState<H, R>,
    ) -> Self
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        Hash: ProverMessage<[H::U]>,
    {
        let mut poly = Vec::with_capacity(r_folded.len() + fresh_padding.len());
        poly.extend_from_slice(r_folded);
        poly.append(&mut fresh_padding);
        let tree_witness = cfg.commit(ps, &[&poly[..]]);
        Self {
            cfg,
            poly,
            tree_witness,
        }
    }

    /// Code-switch blinding input: the full polynomial coefficients.
    fn blinding(&self) -> &[F] {
        &self.poly
    }

    /// Length of the mask oracle polynomial.
    const fn len(&self) -> usize {
        self.poly.len()
    }

    /// Open the cs-mask tree at `covector_region`; returns X_cs = ⟨poly, covector_region⟩.
    /// Zeroizes poly before returning.
    fn prove<H, R>(mut self, covector_region: &[F], ps: &mut ProverState<H, R>) -> F
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        let x_cs = dot(&self.poly, covector_region);
        self.cfg.prove(
            ps,
            self.tree_witness,
            &[&self.poly[..]],
            Some(&[covector_region]),
        );
        self.poly.zeroize();
        x_cs
    }
}

/// Manages all ZK mask oracle state for one round.
///
/// Transitions: Disabled (Standard) or BeforeCodeSwitch → AfterCodeSwitch.
/// The Disabled variant is the Null Object: all methods on it are no-ops or
/// return empty slices, so prove_round has no ZK-specific branches.
enum RoundMaskOracle<'a, F: Field> {
    /// Standard mode or basecase-only round: no mask oracle.
    Disabled,
    /// ZK round — sumcheck-masks tree committed, cs_mask not yet built.
    BeforeCodeSwitch {
        mask_oracle: &'a MaskOracleConfig<F>,
        sc_tree: SumcheckMaskTree<'a, F>,
        cs_fresh_padding: Vec<F>,
    },
    /// ZK round — cs_mask tree committed, ready to discharge.
    AfterCodeSwitch {
        sc_tree: SumcheckMaskTree<'a, F>,
        cs_mask: CodeSwitchMask<'a, F>,
    },
}

impl<'a, F: Field + Default + Zeroize> RoundMaskOracle<'a, F> {
    /// Construct the oracle for this round: Disabled if no mask oracle, otherwise
    /// sample and commit the sumcheck-masks tree (BeforeCodeSwitch).
    fn begin<H, R>(round: &'a RoundConfig<Identity<F>>, ps: &mut ProverState<H, R>) -> Self
    where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        Hash: ProverMessage<[H::U]>,
    {
        let Some(mo) = round.mask_oracle() else {
            return Self::Disabled;
        };
        let num_masks = round.sumcheck().num_rounds();
        let mask_poly_len = match round.sumcheck().mode() {
            SumcheckMode::ZeroKnowledge { mask_length } => mask_length.get(),
            SumcheckMode::Standard => 0,
        };
        let sc_tree =
            SumcheckMaskTree::sample_and_commit(mo.sumcheck_masks(), num_masks, mask_poly_len, ps);
        // cs_fresh_padding is the `s` in cs_mask = (r_folded ‖ s). Sampled here because
        // it does not depend on folding randomness, preserving the Fiat–Shamir RNG ordering.
        let cs_fresh_padding_len = mo.l_zk().get() - round.code_switch().source().mask_length();
        let cs_fresh_padding: Vec<F> = random_vector(ps.rng(), cs_fresh_padding_len);
        Self::BeforeCodeSwitch {
            mask_oracle: mo,
            sc_tree,
            cs_fresh_padding,
        }
    }

    /// Flat sumcheck blinding coefficients. Returns &[] for Disabled.
    fn sumcheck_blinding(&self) -> &[F] {
        match self {
            Self::BeforeCodeSwitch { sc_tree, .. } => sc_tree.blinding(),
            _ => &[],
        }
    }

    /// Build and commit the cs_mask tree, send δ cleartext, reconcile *sum.
    /// Transitions BeforeCodeSwitch → AfterCodeSwitch in place.
    /// No-op for Disabled.
    fn bind_code_switch_mask<H, R>(
        &mut self,
        irs_witness: &IrsWitness<F>,
        opening: &SumcheckOpening<F>,
        sum: &mut F,
        ps: &mut ProverState<H, R>,
    ) where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        Hash: ProverMessage<[H::U]>,
    {
        let (mask_oracle, mut sc_tree, cs_fresh_padding) =
            match std::mem::replace(self, Self::Disabled) {
                Self::BeforeCodeSwitch {
                    mask_oracle,
                    sc_tree,
                    cs_fresh_padding,
                } => (mask_oracle, sc_tree, cs_fresh_padding),
                other => {
                    *self = other;
                    return;
                }
            };

        // Blinding coefficients are no longer needed after sumcheck.
        sc_tree.wipe_blinding();

        // cs_mask = (r_folded ‖ s): r folds the source IRS randomness by the sumcheck challenges.
        let source_mask_len = mask_oracle.l_zk().get() - cs_fresh_padding.len();
        let r_folded = fold_chunks(
            &irs_witness.masks,
            source_mask_len,
            &opening.round_challenges,
        );

        // Build and commit the cs-mask tree (transcript: commitment hash).
        let cs_mask = CodeSwitchMask::build_and_commit(
            &r_folded,
            cs_fresh_padding,
            mask_oracle.cs_mask(),
            ps,
        );

        // Send δ = Σ Mᵢ(rᵢ) cleartext AFTER committing the cs-mask tree (Constr. 9.7 order).
        let delta = sc_tree.eval_sum(&opening.round_challenges);
        ps.prover_message(&delta);

        // Reconcile sum to the unmasked dot product.
        // mask_rlc is Fiat–Shamir; zero has negligible probability for large fields.
        let mask_rlc_inv = opening
            .mask_rlc
            .inverse()
            .expect("mask_rlc non-zero (negligible probability for large fields)");
        *sum = (*sum - delta) * mask_rlc_inv;

        *self = Self::AfterCodeSwitch { sc_tree, cs_mask };
    }

    /// Number of elements to extend the covector by for the ZK region.
    /// Returns 0 for Disabled.
    const fn covector_extension(&self) -> usize {
        match self {
            Self::AfterCodeSwitch { cs_mask, .. } => cs_mask.len(),
            _ => 0,
        }
    }

    /// Code-switch mask polynomial coefficients. Returns &[] for Disabled.
    fn code_switch_blinding(&self) -> &[F] {
        match self {
            Self::AfterCodeSwitch { cs_mask, .. } => cs_mask.blinding(),
            Self::Disabled => &[],
            Self::BeforeCodeSwitch { .. } => {
                debug_assert!(
                    false,
                    "code_switch_blinding called before bind_code_switch_mask"
                );
                &[]
            }
        }
    }

    /// Prove both mask trees and subtract the cs_mask contribution from *sum.
    /// No-op for Disabled.
    fn finish<H, R>(
        self,
        round_challenges: &[F],
        cs_mask_covector: &[F],
        sum: &mut F,
        ps: &mut ProverState<H, R>,
    ) where
        F: Codec<[H::U]>,
        H: DuplexSpongeInterface,
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
        u8: Decoding<[H::U]>,
        [u8; 32]: Decoding<[H::U]>,
        U64: Codec<[H::U]>,
        Hash: ProverMessage<[H::U]>,
    {
        if let Self::AfterCodeSwitch { sc_tree, cs_mask } = self {
            sc_tree.prove(round_challenges, ps);
            *sum -= cs_mask.prove(cs_mask_covector, ps);
        }
    }
}
