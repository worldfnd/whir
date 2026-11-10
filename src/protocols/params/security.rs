//! Security bounds for concrete WHIR and Zook parameters.
//!
//! Formulas follow the [ProveKit whitepaper], sections 5.4, 6.4 and Appendix A.
//! Bounds retain the existing per-step IOP interpretation and grinding policy.
//!
//! [ProveKit whitepaper]: https://github.com/worldfnd/provekit/tree/6605cf5242567e17899a26c27db829c7ce9eb478/whitepaper

use std::f64::consts::{LN_2, LOG2_10};

use ark_ff::Field;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        irs_commit::{Config as IrsConfig, IrsMode},
        params::{
            config::{BasecasePlan, ProtocolConfig, RoundConfig},
            error::{DeriveError, SecurityEvent, ZookSecurityError},
            spec::{DecodingRegime, Mode, SecuritySpec},
        },
        proof_of_work::Config as PowConfig,
    },
};

/// `ρ = 2^-log_inv_rate`. Centralized so the rate formula lives in one place.
pub(super) fn rate(log_inv_rate: f64) -> f64 {
    2_f64.powf(-log_inv_rate)
}

/// Lossy `usize → f64` for analytic-error formulas. Named so individual call
/// sites can stay terse and intent-tagged.
pub(super) const fn usize_to_f64(x: usize) -> f64 {
    x as f64
}

/// log2 of the per-OOD-sample Schwartz–Zippel error: `(k-1)/|F|`.
pub fn ood_per_sample_log2(message_length: usize, field_bits: f64) -> f64 {
    ((message_length - 1) as f64).log2() - field_bits
}

/// `log2(Q - n) = log2(Q) + log2(1 - n/Q)`, where `Q = 2^field_bits`
/// is the field size and `n = domain_size`.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, query phase: the DEEP denominator is `|F \ D| = Q - n`.
pub(super) fn out_of_domain_bits(
    field_bits: f64,
    domain_size: usize,
) -> Result<f64, ZookSecurityError> {
    let fraction = 2.0_f64.powf(usize_to_f64(domain_size).log2() - field_bits);
    if fraction >= 1.0 {
        return Err(ZookSecurityError::EmptyOutOfDomain);
    }
    Ok(field_bits + (-fraction).ln_1p() / LN_2)
}

/// `log2(2^a + 2^b)` for log2 error bounds `a` and `b`.
/// Adding probabilities of equal size costs one bit of security.
pub(super) fn log2_sum(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let larger = a.max(b);
    larger + 2.0_f64.powf(a.min(b) - larger).ln_1p() / LN_2
}

/// Security bits `max(0, -log2(epsilon))` from a log2 error bound.
pub(super) fn error_bits(log_error: f64) -> Bits {
    Bits::new((-log_error).max(0.0))
}

/// Materialized decoding-regime parameters at a known rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodingRegimeParams {
    Unique,
    Johnson { slack: OrderedFloat<f64> },
    Capacity { slack: OrderedFloat<f64> },
}

impl DecodingRegimeParams {
    /// Materialize spec policy at a known rate. Canonical slacks: `√ρ/20` for
    /// Johnson, `ρ/20` for Capacity.
    // TODO: Optimize picking η.
    pub fn from_policy(policy: DecodingRegime, rate: f64) -> Self {
        match policy {
            DecodingRegime::Unique => Self::Unique,
            DecodingRegime::Johnson => Self::Johnson {
                slack: OrderedFloat(rate.sqrt() / 20.0),
            },
            DecodingRegime::Capacity => Self::Capacity {
                slack: OrderedFloat(rate / 20.0),
            },
        }
    }

    pub const fn is_unique(self) -> bool {
        matches!(self, Self::Unique)
    }

    /// `log₂ |Λ(C, δ)|`.
    pub fn list_size_log2(self, log_degree: f64, log_inv_rate: f64) -> f64 {
        match self {
            Self::Unique => 0.0,
            // Johnson: |Λ| = 1 / (2 η √ρ).
            Self::Johnson { slack } => -1.0 - slack.into_inner().log2() + 0.5 * log_inv_rate,
            // Capacity (STIR Conj 5.6): |Λ| = d / (ρ · η).
            Self::Capacity { slack } => log_degree + log_inv_rate - slack.into_inner().log2(),
        }
    }

    /// `|Λ(C, δ)|`.
    pub fn list_size(self, log_degree: f64, log_inv_rate: f64) -> f64 {
        2_f64.powf(self.list_size_log2(log_degree, log_inv_rate))
    }

    /// `log₂(1 − δ)`.
    pub fn one_minus_distance_log2(self, log_inv_rate: f64) -> f64 {
        let one_minus_delta = match self {
            Self::Unique => f64::midpoint(1.0, rate(log_inv_rate)),
            Self::Johnson { slack } => rate(log_inv_rate).sqrt() + slack.into_inner(),
            Self::Capacity { slack } => rate(log_inv_rate) + slack.into_inner(),
        };
        one_minus_delta.log2()
    }

    /// Bits of security delivered by `ood_samples` OOD challenges (STIR Lemma 4.5):
    /// `ood · (|F| − log d) − 2·log|Λ| + 1`. Returns `0` under `Unique`.
    pub fn ood_security_bits(
        self,
        log_degree: f64,
        log_inv_rate: f64,
        field_bits: f64,
        ood_samples: usize,
    ) -> f64 {
        if self.is_unique() {
            return 0.0;
        }
        let log_list = self.list_size_log2(log_degree, log_inv_rate);
        let ood = usize_to_f64(ood_samples);
        ood * (field_bits - log_degree) - 2.0 * log_list + 1.0
    }

    /// `log₂ ε_mca(C, δ)` for the per-step proximity-gaps error.
    ///
    /// - Unique: `(k − 1) / |F|`, log = `log k − |F|` (with `+ log ρ⁻¹`).
    /// - Johnson: BCSS25 Theorem 1.5 at canonical `η = √ρ/20`, `m = 10`:
    ///   `ε ≈ (2·10.5⁵/3) · n · ρ^{−3/2} / |F|`.
    /// - Capacity: STIR Conj 5.6, `ε ≈ d / (η · ρ²) / |F|`.
    pub fn eps_mca_log2(self, log_inv_rate: f64, message_length: usize, field_bits: f64) -> f64 {
        let log_k = usize_to_f64(message_length).log2();
        let error = match self {
            Self::Unique => log_k + log_inv_rate,
            Self::Johnson { slack } => {
                assert!(
                    slack.into_inner().log2() >= -(0.5 * log_inv_rate + LOG2_10 + 1.0) - 1e-6,
                    "Johnson slack η below BCSS25 lower bound; ε_mca would be over-optimistic",
                );
                // BCSS25 with m = 10: log_2(2·10.5⁵/3) + log n + 1.5·log ρ⁻¹.
                // Substituting n = k/ρ (codeword length) gives the `log_k`
                // (message length) + 2.5·log ρ⁻¹ form below.
                let bcss25_const = (2.0 * 10.5_f64.powi(5) / 3.0).log2();
                bcss25_const + log_k + 2.5 * log_inv_rate
            }
            Self::Capacity { slack } => {
                // η lower bound from STIR Conj 5.6; see the Johnson arm for why
                // this is a hard assert rather than debug_assert.
                assert!(
                    slack.into_inner().log2() >= -(log_inv_rate + LOG2_10 + 1.0) - 1e-6,
                    "Capacity slack η below STIR Conj 5.6 lower bound; ε_mca would be over-optimistic",
                );
                // d / (η · ρ²) at canonical η = ρ/20: log d + log 20 + 3·log ρ⁻¹.
                log_k + 3.0 * log_inv_rate + LOG2_10 + 1.0
            }
        };
        error - field_bits
    }
}

/// Constrained correlated-agreement error for constraints of degree at most two:
/// `epsilon <= L * ((2/3) * L^4 * rho + 1) * n / Q`.
/// Here `L = code.list_size()`, `rho = code.rate()`, `n = code.codeword_length()`,
/// and `Q` is the size of the target field where challenges are sampled.
///
/// ProveKit Prover Whitepaper Appendix A, “Correlated agreement”, constrained
/// correlated-agreement theorem.
/// Its factor `1 - delta = (m - 1)/n`, for code dimension `m`, is conservatively
/// replaced by `rho = m/n`. The same bound covers degree-one batching.
pub(super) fn constrained_agreement_bits<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<Bits, ZookSecurityError> {
    validate_zook_code(code)?;
    let list_bits = code.list_size().log2();
    let log_factor = (2.0_f64 / 3.0).log2() + 4.0 * list_bits + code.rate().log2();
    Ok(error_bits(
        list_bits + log2_sum(log_factor, 0.0) + usize_to_f64(code.codeword_length()).log2()
            - M::Target::field_size_bits(),
    ))
}

/// Require degree-preserving encoding and `delta/2 <= theta < 1 - sqrt(1-delta)`.
/// Here `delta = (n - m + 1)/n` is the relative Reed–Solomon distance,
/// `n = codeword_length()`, `m = message_length()`, and `theta` is the
/// configured proximity radius.
///
/// ProveKit Prover Whitepaper Appendix A, “Correlated agreement”, constrained
/// correlated-agreement theorem. Section 5.3, “Overlaid masks instead of
/// zk-encoding”, defines the degree-preserving commitment to `p + mask`.
pub(super) fn validate_zook_code<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<(), ZookSecurityError> {
    if !matches!(code.mode(), IrsMode::Standard) {
        return Err(ZookSecurityError::AppendedMasks);
    }
    let DecodingRegimeParams::Johnson { .. } = code.regime() else {
        return Err(ZookSecurityError::UnsupportedRegime);
    };
    let rho = code.rate();
    let theta = 1.0 - 2.0_f64.powf(code.regime().one_minus_distance_log2(-rho.log2()));
    // RS minimum distance is (n - k + 1) / n, including the constant term.
    let distance = 1.0 - rho + 1.0 / usize_to_f64(code.codeword_length());
    if theta < distance / 2.0 || theta >= 1.0 - (1.0 - distance).sqrt() {
        return Err(ZookSecurityError::InvalidProximity);
    }
    Ok(())
}

/// Restrict the final witness bound to one polynomial per commitment.
pub(super) fn validate_single_polynomial<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<(), ZookSecurityError> {
    validate_zook_code(code)?;
    if code.num_vectors() != 1 || code.interleaving_depth() != 1 {
        return Err(ZookSecurityError::ExpectedSinglePolynomial);
    }
    Ok(())
}

/// Require two full-length columns: joint masks and their final-opening blinder.
pub(super) fn validate_joint_mask_code<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<(), ZookSecurityError> {
    validate_zook_code(code)?;
    if code.num_vectors() != 2 || code.interleaving_depth() != 1 {
        return Err(ZookSecurityError::ExpectedMaskAndBlinder);
    }
    Ok(())
}

/// Per-step security of the configured sumcheck protocol.
///
/// Zook uses the constrained correlated-agreement bound after joint-mask binding.
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, sumcheck phase; Appendix A gives the agreement bound.
/// The degree-two bound also covers degree-one mask batching.
/// This is a bound for one step, not a union over all steps.
pub fn sumcheck_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    is_zk: bool,
) -> Result<Bits, ZookSecurityError> {
    if is_zk {
        return constrained_agreement_bits(source);
    }
    let polynomial_identity = M::Target::field_size_bits() - source.list_size().log2() - 1.0;
    Ok(Bits::new(
        source
            .rbr_soundness_fold_prox_gaps()
            .min(polynomial_identity)
            .max(0.0),
    ))
}

/// A quadratic sumcheck step on a revealed vector fails with probability at most `2 / |F|`.
///
/// The vector is fixed before these challenges, so there is no codeword-list factor.
/// This is the additional reduction after §5.4, “The final opening protocol”.
pub(super) fn revealed_sumcheck_error_bits<F: Field>() -> Bits {
    Bits::new((F::field_size_bits() - 1.0).max(0.0))
}

/// Per-step query and aggregation security.
pub fn code_switch_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
    t_ood: usize,
    is_zk: bool,
) -> Result<Bits, DeriveError> {
    if is_zk {
        let (queries, aggregation) = zook_query_error_bits(source, target)?;
        return Ok(queries.min(aggregation));
    }
    let field_bits = M::Target::field_size_bits();
    let list = target.list_size();
    let degree = source.message_length();
    let ood = usize_to_f64(t_ood) * (field_bits - usize_to_f64(degree.saturating_sub(1)).log2())
        - (list * (list - 1.0) / 2.0).log2();
    let count = source
        .in_domain_samples()
        .checked_mul(source.interleaving_depth())
        .and_then(|count| count.checked_add(t_ood))
        .ok_or(DeriveError::SizeOverflow)?;
    let combination = field_bits - usize_to_f64(count).log2() - list.log2();
    Ok(Bits::new(
        ood.min(source.rbr_queries()).min(combination).max(0.0),
    ))
}

/// Query and aggregation security, conditional on joint-mask binding:
/// `epsilon_query <= L^2 * m / (2*(Q - n)) + (1 - theta)^s`;
/// `epsilon_aggregation <= L * s / Q`.
/// Here `L = target.list_size()`, `m = target.vector_size()`,
/// `n = target.codeword_length()`, and `Q` is the target-field size.
/// The source has proximity radius `theta` and `s = source.in_domain_samples()`.
/// The implementation samples DEEP challenges from the full field.
/// The denominator `Q - n` retains the whitepaper's conservative root-count bound.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, query phase and aggregation step.
/// The query error is the sum of DEEP and in-domain errors.
pub(super) fn zook_query_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
) -> Result<(Bits, Bits), ZookSecurityError> {
    validate_zook_code(source)?;
    validate_zook_code(target)?;
    if source.num_vectors() != 1
        || target.num_vectors() != 1
        || source.message_length() != target.vector_size()
    {
        return Err(ZookSecurityError::InvalidRoundShape);
    }
    let field_bits = M::Target::field_size_bits();
    let list_bits = target.list_size().log2();
    let outside_bits = out_of_domain_bits(field_bits, target.codeword_length())?;
    // The DEEP evaluation is of the whole folded polynomial, before its
    // coefficients are split into the target's interleaved columns.
    let deep_log_error =
        2.0 * list_bits - 1.0 + usize_to_f64(target.vector_size()).log2() - outside_bits;
    let query_log_error = -source.rbr_queries();
    Ok((
        error_bits(log2_sum(deep_log_error, query_log_error)),
        error_bits(list_bits + usize_to_f64(source.in_domain_samples()).log2() - field_bits),
    ))
}

/// Final-opening security conditional on joint-mask binding.
///
/// `epsilon_combination = max(A_witness, A_masks)`;
/// `epsilon_queries <= (1 - theta_witness)^s_witness + (1 - theta_masks)^s_masks`.
/// Each `A` bounds degree-one constrained correlated agreement for its code;
/// each `theta` is that code's proximity radius and each `s` its query count.
/// The mask commitment holds two full-length columns combined with weights
/// `[lambda, 1]`; sharing their root leaves the degree-one combination unchanged.
/// The implementation conservatively uses the degree-two agreement bound.
/// Taking the maximum of error probabilities gives the minimum of security bits.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “ZO0K soundness under
/// mask binding”, final-opening batching and query bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ZookFinalOpeningSoundness {
    /// One common random combination for the witness and joint mask commitments.
    pub combination: Bits,
    /// Sum of the two independent code-query error bounds.
    pub queries: Bits,
}

impl ZookFinalOpeningSoundness {
    /// Evaluate the final witness code and joint mask code at their actual query counts.
    pub fn for_codes<F: Field>(
        witness: &IrsConfig<Identity<F>>,
        masks: &IrsConfig<Identity<F>>,
    ) -> Result<Self, ZookSecurityError> {
        validate_single_polynomial(witness)?;
        validate_joint_mask_code(masks)?;
        Ok(Self {
            combination: constrained_agreement_bits(witness)?
                .min(constrained_agreement_bits(masks)?),
            queries: error_bits(log2_sum(-witness.rbr_queries(), -masks.rbr_queries())),
        })
    }
}

/// Joint-mask DEEP binding plus the initial-mask Frobenius check.
///
/// `epsilon <= L*(L-1)*m/(2*(Q-1)) + L*m/(Q-1) = L*(L+1)*m/(2*(Q-1))`.
/// Here `L = masks.list_size()`, `m = masks.message_length()`, and `Q = |F|`.
/// The first term bounds DEEP collisions; the second bounds the Frobenius check.
/// Both concern only the joint-mask column; the other column is its final-opening blinder.
/// Unlike §6.3, “The zk-IOPP”, Phase 0, sampling excludes zero, where the
/// shifted membership blinder vanishes. Nonzero evaluation-domain points remain
/// allowed, so the root-count bounds use the challenge-domain size `Q-1`.
///
/// ProveKit Prover Whitepaper §6.4, “Soundness”, theorem “IOPP soundness”,
/// Phase 0 bound and the following union-bound remark.
pub fn zook_mask_binding_bits<F: Field>(
    masks: &IrsConfig<Identity<F>>,
) -> Result<Bits, ZookSecurityError> {
    validate_joint_mask_code(masks)?;
    let list = masks.list_size();
    Ok(error_bits(
        list.log2() + (list + 1.0).log2() - 1.0 + usize_to_f64(masks.message_length()).log2()
            - out_of_domain_bits(F::field_size_bits(), 1)?,
    ))
}

/// One recomputed security bound and its configured proof of work.
#[derive(Clone, Copy, Debug)]
pub struct SecurityCheck {
    pub kind: SecurityEvent,
    pub pow: PowConfig,
    pub analytic: Bits,
}

impl<M: Embedding> ProtocolConfig<M> {
    /// Recompute the parameter selector's per-step security bounds.
    pub fn security_report(&self) -> Result<Vec<SecurityCheck>, DeriveError> {
        let mut slots = Vec::new();
        for report in self
            .first_round()
            .into_iter()
            .map(|round| round.security_report())
            .chain(self.tail_rounds().map(|round| round.security_report()))
        {
            slots.extend(report?);
        }
        match self.basecase() {
            BasecasePlan::Standard { config } => slots.push(SecurityCheck {
                kind: SecurityEvent::BasecaseSumcheck,
                pow: config.sumcheck().round_pow(),
                analytic: sumcheck_error_bits(config.commit(), false)?,
            }),
            BasecasePlan::ZeroKnowledge {
                commit, sumcheck, ..
            } => {
                let joint = self.joint_masks().ok_or(DeriveError::InvalidMaskLayout)?;
                let opening = ZookFinalOpeningSoundness::for_codes(commit, &joint.commit)?;
                slots.extend([
                    SecurityCheck {
                        kind: SecurityEvent::BasecaseSumcheck,
                        pow: sumcheck.round_pow(),
                        analytic: revealed_sumcheck_error_bits::<M::Target>(),
                    },
                    SecurityCheck {
                        kind: SecurityEvent::JointMaskBinding,
                        pow: PowConfig::none(),
                        analytic: zook_mask_binding_bits(&joint.commit)?,
                    },
                    SecurityCheck {
                        kind: SecurityEvent::FinalOpeningCombination,
                        pow: PowConfig::none(),
                        analytic: opening.combination,
                    },
                    SecurityCheck {
                        kind: SecurityEvent::FinalOpeningQueries,
                        pow: PowConfig::none(),
                        analytic: opening.queries,
                    },
                ]);
            }
        }
        Ok(slots)
    }

    pub(super) fn validate_security(&self) -> Result<(), DeriveError> {
        let eps = Bits::new(1e-9);
        let target = Bits::new(f64::from(self.security().target_security_bits));
        let max = Bits::new(f64::from(self.security().pow_budget.bits()));
        for check in self.security_report()? {
            let required = check.pow.difficulty();
            if required > max {
                return Err(DeriveError::PowBudgetExceeded {
                    pow: check.kind,
                    required,
                    max,
                });
            }
            let pow_bits = if self.security().mode.is_zk() {
                Bits::new(0.0)
            } else {
                required
            };
            if check.analytic + pow_bits + eps < target {
                return Err(DeriveError::SecurityTargetNotMet {
                    pow: check.kind,
                    analytic: check.analytic,
                    pow_bits,
                    target,
                });
            }
        }
        Ok(())
    }

    /// Minimum per-step security; excludes grinding credit.
    pub fn analytic_bits(&self) -> Bits {
        self.security_report()
            .expect("valid protocol security bounds")
            .into_iter()
            .map(|check| check.analytic)
            .min()
            .expect("basecase supplies a security bound")
    }
}

impl<M: Embedding> RoundConfig<'_, M> {
    pub fn analytic_bits(&self) -> Bits {
        self.security_report()
            .expect("valid round security bounds")
            .into_iter()
            .map(|check| check.analytic)
            .min()
            .expect("round supplies security bounds")
    }

    fn security_report(&self) -> Result<Vec<SecurityCheck>, DeriveError> {
        let cs = self.code_switch();
        let is_zk = self.masks().is_some();
        let sumcheck = sumcheck_error_bits(cs.source(), is_zk)?;
        let mut slots = vec![
            SecurityCheck {
                kind: SecurityEvent::RoundSumcheck {
                    index: self.round_index(),
                },
                pow: self.sumcheck().round_pow(),
                analytic: sumcheck,
            },
            SecurityCheck {
                kind: SecurityEvent::RoundCodeSwitch {
                    index: self.round_index(),
                },
                pow: cs.pow(),
                analytic: code_switch_error_bits(
                    cs.source(),
                    cs.target(),
                    cs.out_domain_samples(),
                    is_zk,
                )?,
            },
        ];
        if is_zk {
            slots.push(SecurityCheck {
                kind: SecurityEvent::RoundMaskCombination {
                    index: self.round_index(),
                },
                pow: PowConfig::none(),
                analytic: sumcheck,
            });
        }
        Ok(slots)
    }
}

/// Enforce the target, allowing grinding credit only in Standard mode.
///
/// Zook requires the full analytic target until a grinding reduction is supplied.
pub(super) fn grind_to_at(
    spec: &SecuritySpec,
    analytic: Bits,
    pow_kind: SecurityEvent,
) -> Result<PowConfig, DeriveError> {
    let target = Bits::new(f64::from(spec.target_security_bits));
    if matches!(spec.mode, Mode::ZeroKnowledge) {
        if analytic < target {
            return Err(DeriveError::SecurityTargetNotMet {
                pow: pow_kind,
                analytic,
                pow_bits: Bits::new(0.0),
                target,
            });
        }
        return Ok(PowConfig::none());
    }
    let mut pow = PowConfig::grind_to(target, analytic, spec.hash_id).map_err(|source| {
        DeriveError::PowUngrindable {
            pow: pow_kind,
            source,
        }
    })?;
    // grind_to rounds the threshold upward. Tighten a fractional gap by one
    // threshold unit instead of accepting less than the configured target.
    if analytic + pow.difficulty() < target {
        pow.threshold = pow.threshold.saturating_sub(1);
    }
    let required = pow.difficulty();
    let max = Bits::new(f64::from(spec.pow_budget.bits()));
    if required > max {
        return Err(DeriveError::PowBudgetExceeded {
            pow: pow_kind,
            required,
            max,
        });
    }
    Ok(pow)
}

#[cfg(test)]
#[path = "tests/security.rs"]
mod tests;
