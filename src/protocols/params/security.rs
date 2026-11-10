//! WHIR/ZOOK bounds from ProveKit §6.4 and Appendix A, pinned at 6605cf5242567e17899a26c27db829c7ce9eb478.
use std::f64::consts::{LN_2, LOG2_10};

use ark_ff::Field;

use super::{require, DecodingRegime, DeriveError, Params, Result};
use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        irs_commit::{Config as Irs, IrsMode},
        proof_of_work::Config as Pow,
    },
};
type Target<M> = Irs<Identity<<M as Embedding>::Target>>;

/// Raw-rate slack and the original log/exp roundtrip give log2(1-distance).
pub fn decoding_distance(policy: DecodingRegime, rho: f64) -> f64 {
    let reconstructed = rate(-rho.log2());
    match policy {
        DecodingRegime::Unique => f64::midpoint(1.0, reconstructed),
        DecodingRegime::Johnson => reconstructed.sqrt() + rho.sqrt() / 20.0,
        DecodingRegime::Capacity => reconstructed + rho / 20.0,
    }
    .log2()
}

/// Canonical [list size, log2(1-distance), log2(MCA error)].
/// Johnson uses BCSS25 Theorem 1.5; Capacity uses STIR Conjecture 5.6.
pub fn decoding(policy: DecodingRegime, rho: f64, length: usize, field: f64) -> [f64; 3] {
    use DecodingRegime::{Capacity, Johnson, Unique};
    let r = -rho.log2();
    let d = (length as f64).log2();
    let slack = if policy == Johnson {
        rho.sqrt() / 20.0
    } else {
        rho / 20.0
    };
    let list = match policy {
        Unique => 0.0,
        Johnson => -1.0 - slack.log2() + 0.5 * r,
        Capacity => d + r - slack.log2(),
    };
    let mca = match policy {
        Unique => d + r,
        Johnson => {
            assert!(slack.log2() >= -(0.5 * r + LOG2_10 + 1.0) - 1e-6);
            (2.0 * 10.5_f64.powi(5) / 3.0).log2() + d + 2.5 * r
        }
        Capacity => {
            assert!(slack.log2() >= -(r + LOG2_10 + 1.0) - 1e-6);
            d + 3.0 * r + LOG2_10 + 1.0
        }
    };
    [
        2_f64.powf(list),
        decoding_distance(policy, rho),
        mca - field,
    ]
}

pub(super) fn rate(r: f64) -> f64 {
    2_f64.powf(-r)
}
pub(super) const fn usize_to_f64(n: usize) -> f64 {
    n as f64
}
pub fn ood_per_sample_log2(k: usize, field: f64) -> f64 {
    ((k - 1) as f64).log2() - field
}
fn bits(error: f64) -> Bits {
    Bits::new((-error).max(0.0))
}

/// Stable log2 sums and challenge domains preserve small-error precision.
pub(super) fn log2_sum(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let larger = a.max(b);
    larger + 2_f64.powf(a.min(b) - larger).ln_1p() / LN_2
}
pub(super) fn out_of_domain_bits(field: f64, n: usize) -> Result<f64> {
    let fraction = 2_f64.powf((n as f64).log2() - field);
    let exhausts = fraction >= 1.0;
    require(!exhausts, "code domain exhausts the challenge field")?;
    Ok(field + (-fraction).ln_1p() / LN_2)
}

/// §5.3 preserves degree; Appendix A requires delta/2 <= theta < 1-sqrt(1-delta).
pub(super) fn validate_zook_code<M: Embedding>(
    code: &Irs<M>,
    columns: Option<usize>,
) -> Result<()> {
    require(
        matches!(code.mode(), IrsMode::Standard) && code.regime() == DecodingRegime::Johnson,
        "Zook requires standard IRS encoding and Johnson decoding",
    )?;
    let rho = code.rate();
    let theta = 1.0 - 2_f64.powf(decoding_distance(code.regime(), rho));
    let distance = 1.0 - rho + 1.0 / code.codeword_length() as f64;
    require(
        !(theta < distance / 2.0 || theta >= 1.0 - (1.0 - distance).sqrt()),
        "invalid Zook proximity",
    )?;
    require(
        columns.is_none_or(|n| code.num_vectors() == n && code.interleaving_depth() == 1),
        "invalid Zook columns",
    )
}

/// Appendix A: L*((2/3)*L^4*rho+1)*n/Q; also bounds degree-one batching.
fn agreement<M: Embedding>(code: &Irs<M>) -> Bits {
    let list = code.list_size().log2();
    let factor = (2.0_f64 / 3.0).log2() + 4.0 * list + code.rate().log2();
    bits(
        list + log2_sum(factor, 0.0) + (code.codeword_length() as f64).log2()
            - M::Target::field_size_bits(),
    )
}

/// §6.4 queries: L²*m/(2*(Q-n))+(1-theta)^s; aggregation: L*s/Q.
pub(super) fn zook_query_error_bits<M: Embedding>(
    source: &Irs<M>,
    target: &Target<M>,
) -> Result<(Bits, Bits)> {
    validate_zook_code(source, None)?;
    validate_zook_code(target, None)?;
    require(
        source.num_vectors() == 1
            && target.num_vectors() == 1
            && source.message_length() == target.vector_size(),
        "invalid round dimensions",
    )?;
    let field = M::Target::field_size_bits();
    let list = target.list_size().log2();
    let outside = out_of_domain_bits(field, target.codeword_length())?;
    let deep = 2.0 * list - 1.0 + (target.vector_size() as f64).log2() - outside;
    Ok((
        bits(log2_sum(deep, -source.rbr_queries())),
        bits(list + (source.in_domain_samples() as f64).log2() - field),
    ))
}

/// Per-step sumcheck and optional code-switch bounds. Final Zook sumcheck uses 2/Q.
pub(super) fn round_bits<M: Embedding>(
    source: &Irs<M>,
    target: Option<&Target<M>>,
    t: usize,
    zk: bool,
) -> Result<(Bits, Option<Bits>)> {
    let field = M::Target::field_size_bits();
    let sumcheck = if zk {
        validate_zook_code(source, target.is_none().then_some(1))?;
        if target.is_some() {
            agreement(source)
        } else {
            Bits::new((field - 1.0).max(0.0))
        }
    } else {
        let identity = field - source.list_size().log2() - 1.0;
        Bits::new(source.rbr_soundness_fold_prox_gaps().min(identity).max(0.0))
    };
    let Some(target) = target else {
        return Ok((sumcheck, None));
    };
    let switch = if zk {
        let (queries, aggregation) = zook_query_error_bits(source, target)?;
        queries.min(aggregation)
    } else {
        let list = target.list_size();
        let ood = t as f64 * (field - (source.message_length().saturating_sub(1) as f64).log2())
            - (list * (list - 1.0) / 2.0).log2();
        let count = source
            .in_domain_samples()
            .checked_mul(source.interleaving_depth())
            .and_then(|n| n.checked_add(t))
            .ok_or(DeriveError("query count overflow"))?;
        let combination = field - (count as f64).log2() - list.log2();
        Bits::new(ood.min(source.rbr_queries()).min(combination).max(0.0))
    };
    Ok((sumcheck, Some(switch)))
}

/// §6.4 [binding, final combination, final queries]; binding includes Frobenius error.
/// Errors: L*(L+1)*m/(2*(Q-1)), max(A_witness,A_masks), and summed query errors.
pub(super) fn final_bits<F: Field>(
    witness: &Irs<Identity<F>>,
    masks: &Irs<Identity<F>>,
) -> Result<(Bits, Bits, Bits)> {
    validate_zook_code(witness, Some(1))?;
    validate_zook_code(masks, Some(2))?;
    let list = masks.list_size();
    let binding = bits(
        list.log2() + (list + 1.0).log2() - 1.0 + (masks.message_length() as f64).log2()
            - out_of_domain_bits(F::field_size_bits(), 1)?,
    );
    Ok((
        binding,
        agreement(witness).min(agreement(masks)),
        bits(log2_sum(-witness.rbr_queries(), -masks.rbr_queries())),
    ))
}

/// Zook gets no grinding credit. Tighten rounded thresholds before checking the budget.
pub(super) fn grind_to_at(spec: &Params, analytic: Bits, label: &'static str) -> Result<Pow> {
    let target = Bits::new(f64::from(spec.bits));
    if spec.zk {
        require(analytic >= target, label)?;
        return Ok(Pow::none());
    }
    let mut pow = Pow::grind_to(target, analytic, spec.hash).map_err(|_| DeriveError(label))?;
    if analytic + pow.difficulty() < target {
        pow.threshold = pow.threshold.saturating_sub(1);
    }
    require(pow.difficulty() <= Bits::new(f64::from(spec.pow)), label)?;
    Ok(pow)
}
