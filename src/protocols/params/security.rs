//! Executed WHIR and Zook security bounds, with unchanged per-step IOP accounting.
//! Whitepaper: <https://github.com/worldfnd/provekit/tree/6605cf5242567e17899a26c27db829c7ce9eb478/whitepaper>

use std::f64::consts::{LN_2, LOG2_10};

use ark_ff::Field;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use super::{DecodingRegime, DeriveError, Mode, SecuritySpec};
use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        irs_commit::{Config as IrsConfig, IrsMode},
        proof_of_work::Config as PowConfig,
    },
};

/// `rho = 2^-log_inv_rate`.
pub(super) fn rate(log_inv_rate: f64) -> f64 {
    2_f64.powf(-log_inv_rate)
}

pub(super) const fn usize_to_f64(x: usize) -> f64 {
    x as f64
}

/// Per-sample Schwartz–Zippel error: `log2((k-1)/|F|)`.
pub fn ood_per_sample_log2(message_length: usize, field_bits: f64) -> f64 {
    ((message_length - 1) as f64).log2() - field_bits
}

/// Whitepaper §6.4, “ZO0K soundness under mask binding”, query phase: `log2(Q-n)`.
pub(super) fn out_of_domain_bits(field_bits: f64, domain_size: usize) -> Result<f64, DeriveError> {
    let fraction = 2.0_f64.powf(usize_to_f64(domain_size).log2() - field_bits);
    if fraction >= 1.0 {
        return Err(DeriveError::Invalid(
            "code domain exhausts the challenge field",
        ));
    }
    Ok(field_bits + (-fraction).ln_1p() / LN_2)
}

/// Stable `log2(2^a + 2^b)` for the sum of two error probabilities.
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

/// Security bits: `max(0, -log2(epsilon))`.
pub(super) fn error_bits(log_error: f64) -> Bits {
    Bits::new((-log_error).max(0.0))
}

/// Decoding policy with its materialized proximity slack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodingRegimeParams {
    Unique,
    Johnson { slack: OrderedFloat<f64> },
    Capacity { slack: OrderedFloat<f64> },
}

impl DecodingRegimeParams {
    /// Canonical slack: `sqrt(rho)/20` for Johnson; `rho/20` for Capacity.
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

    /// Johnson: `L=1/(2*eta*sqrt(rho))`; STIR Conjecture 5.6: `L=d/(rho*eta)`.
    pub fn list_size_log2(self, log_degree: f64, log_inv_rate: f64) -> f64 {
        match self {
            Self::Unique => 0.0,
            // Johnson: |Λ| = 1 / (2 η √ρ).
            Self::Johnson { slack } => -1.0 - slack.into_inner().log2() + 0.5 * log_inv_rate,
            // Capacity (STIR Conj 5.6): |Λ| = d / (ρ · η).
            Self::Capacity { slack } => log_degree + log_inv_rate - slack.into_inner().log2(),
        }
    }

    pub fn list_size(self, log_degree: f64, log_inv_rate: f64) -> f64 {
        2_f64.powf(self.list_size_log2(log_degree, log_inv_rate))
    }

    /// `log2(1-delta)` for the configured decoding radius.
    pub fn one_minus_distance_log2(self, log_inv_rate: f64) -> f64 {
        let one_minus_delta = match self {
            Self::Unique => f64::midpoint(1.0, rate(log_inv_rate)),
            Self::Johnson { slack } => rate(log_inv_rate).sqrt() + slack.into_inner(),
            Self::Capacity { slack } => rate(log_inv_rate) + slack.into_inner(),
        };
        one_minus_delta.log2()
    }

    /// BCSS25 Theorem 1.5 (Johnson, m=10); STIR Conjecture 5.6 (Capacity).
    pub fn eps_mca_log2(self, log_inv_rate: f64, message_length: usize, field_bits: f64) -> f64 {
        let log_k = usize_to_f64(message_length).log2();
        let error = match self {
            Self::Unique => log_k + log_inv_rate,
            Self::Johnson { slack } => {
                assert!(
                    slack.into_inner().log2() >= -(0.5 * log_inv_rate + LOG2_10 + 1.0) - 1e-6,
                    "Johnson slack η below BCSS25 lower bound; ε_mca would be over-optimistic",
                );
                // Substitute n=k/rho in BCSS25's n*rho^(-3/2) bound.
                let bcss25_const = (2.0 * 10.5_f64.powi(5) / 3.0).log2();
                bcss25_const + log_k + 2.5 * log_inv_rate
            }
            Self::Capacity { slack } => {
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

/// Whitepaper Appendix A, constrained correlated agreement: `L*((2/3)*L^4*rho+1)*n/Q`.
/// The degree-two bound also covers degree-one batching; `rho` bounds `(m-1)/n`.
pub(super) fn constrained_agreement_bits<M: Embedding>(
    code: &IrsConfig<M>,
) -> Result<Bits, DeriveError> {
    validate_zook_code(code, None)?;
    let list_bits = code.list_size().log2();
    let log_factor = (2.0_f64 / 3.0).log2() + 4.0 * list_bits + code.rate().log2();
    Ok(error_bits(
        list_bits + log2_sum(log_factor, 0.0) + usize_to_f64(code.codeword_length()).log2()
            - M::Target::field_size_bits(),
    ))
}

/// Appendix A requires `delta/2 <= theta < 1-sqrt(1-delta)`.
/// Whitepaper §5.3, “Overlaid masks instead of zk-encoding”, preserves the code degree.
pub(super) fn validate_zook_code<M: Embedding>(
    code: &IrsConfig<M>,
    columns: Option<usize>,
) -> Result<(), DeriveError> {
    if !matches!(code.mode(), IrsMode::Standard) {
        return Err(DeriveError::Invalid(
            "Zook bounds require standard IRS encoding",
        ));
    }
    let DecodingRegimeParams::Johnson { .. } = code.regime() else {
        return Err(DeriveError::Invalid("Zook bounds require Johnson decoding"));
    };
    let rho = code.rate();
    let theta = 1.0 - 2.0_f64.powf(code.regime().one_minus_distance_log2(-rho.log2()));
    // RS minimum distance is (n - k + 1) / n, including the constant term.
    let distance = 1.0 - rho + 1.0 / usize_to_f64(code.codeword_length());
    if theta < distance / 2.0 || theta >= 1.0 - (1.0 - distance).sqrt() {
        return Err(DeriveError::Invalid(
            "proximity parameter is outside the correlated-agreement interval",
        ));
    }
    if columns.is_some_and(|n| code.num_vectors() != n || code.interleaving_depth() != 1) {
        return Err(DeriveError::Invalid("unexpected Zook code dimensions"));
    }
    Ok(())
}

/// Whitepaper §6.4, “ZO0K soundness under mask binding”, sumcheck phase.
/// Zook uses constrained agreement per step, conditional on joint-mask binding.
pub fn sumcheck_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    is_zk: bool,
) -> Result<Bits, DeriveError> {
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

/// Revealed-vector sumcheck: error `2/|F|`, after §5.4, “The final opening protocol”.
pub(super) fn revealed_sumcheck_error_bits<F: Field>() -> Bits {
    Bits::new((F::field_size_bits() - 1.0).max(0.0))
}

/// Per-step query and aggregation security; Standard permits repeated DEEP samples.
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

/// Whitepaper §6.4, “ZO0K soundness under mask binding”, query and aggregation phases.
/// Errors: `L^2*m/(2*(Q-n)) + (1-theta)^s` and `L*s/Q`.
/// DEEP samples use the full field; `Q-n` retains the conservative whitepaper denominator.
pub(super) fn zook_query_error_bits<M: Embedding>(
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
) -> Result<(Bits, Bits), DeriveError> {
    validate_zook_code(source, None)?;
    validate_zook_code(target, None)?;
    if source.num_vectors() != 1
        || target.num_vectors() != 1
        || source.message_length() != target.vector_size()
    {
        return Err(DeriveError::Invalid(
            "target vector size must equal source polynomial length",
        ));
    }
    let field_bits = M::Target::field_size_bits();
    let list_bits = target.list_size().log2();
    let outside_bits = out_of_domain_bits(field_bits, target.codeword_length())?;
    // DEEP evaluates the whole folded polynomial, before target interleaving.
    let deep_log_error =
        2.0 * list_bits - 1.0 + usize_to_f64(target.vector_size()).log2() - outside_bits;
    let query_log_error = -source.rbr_queries();
    Ok((
        error_bits(log2_sum(deep_log_error, query_log_error)),
        error_bits(list_bits + usize_to_f64(source.in_domain_samples()).log2() - field_bits),
    ))
}

/// Whitepaper §6.4, “ZO0K soundness under mask binding”, final-opening bounds.
/// Combination error: `max(A_witness,A_masks)`; queries add their independent errors.
/// Each A uses the degree-two agreement bound, conservatively covering degree one.
pub(super) fn zook_final_opening_bits<F: Field>(
    witness: &IrsConfig<Identity<F>>,
    masks: &IrsConfig<Identity<F>>,
) -> Result<(Bits, Bits), DeriveError> {
    validate_zook_code(witness, Some(1))?;
    validate_zook_code(masks, Some(2))?;
    Ok((
        constrained_agreement_bits(witness)?.min(constrained_agreement_bits(masks)?),
        error_bits(log2_sum(-witness.rbr_queries(), -masks.rbr_queries())),
    ))
}

/// Whitepaper §6.4, “IOPP soundness”, Phase 0 and its union-bound remark.
/// DEEP binding plus Frobenius error: `L*(L+1)*m/(2*(Q-1))`.
/// Sampling excludes zero because the shifted membership blinder vanishes there.
pub fn zook_mask_binding_bits<F: Field>(
    masks: &IrsConfig<Identity<F>>,
) -> Result<Bits, DeriveError> {
    validate_zook_code(masks, Some(2))?;
    let list = masks.list_size();
    Ok(error_bits(
        list.log2() + (list + 1.0).log2() - 1.0 + usize_to_f64(masks.message_length()).log2()
            - out_of_domain_bits(F::field_size_bits(), 1)?,
    ))
}

/// Enforce the full analytic target for Zook; permit grinding credit only for Standard.
pub(super) fn grind_to_at(
    spec: &SecuritySpec,
    analytic: Bits,
    pow_kind: &'static str,
) -> Result<PowConfig, DeriveError> {
    let target = Bits::new(f64::from(spec.target_security_bits));
    if matches!(spec.mode, Mode::ZeroKnowledge) {
        if analytic < target {
            return Err(DeriveError::SecurityTargetNotMet {
                pow: pow_kind,
                analytic,
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
    let max = Bits::new(f64::from(spec.pow_budget));
    if required > max {
        return Err(DeriveError::PowBudgetExceeded {
            pow: pow_kind,
            required,
            max,
        });
    }
    Ok(pow)
}
