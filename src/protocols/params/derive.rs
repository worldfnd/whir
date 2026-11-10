//! Construct commitment dimensions, round bounds, and masks before proving.

use thiserror::Error;

use super::{
    embedding_degree,
    masks::{mask_coefficients, solve as solve_joint_masks, WitnessMaskConfig},
    search::plan_adaptive_rates,
    security::{
        code_switch_error_bits, grind_to_at, rate, revealed_sumcheck_error_bits,
        sumcheck_error_bits, usize_to_f64, validate_zook_code,
    },
    DecodingRegime, Mode, Node, ProtocolConfig, ProtocolSpec, RateSchedule, RoundConfig,
    RoundContext, SecuritySpec, TuningSpec,
};
use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        irs_commit::{CodewordLengthError, Config as IrsConfig, IrsMode, IrsParams},
        proof_of_work::PowError,
    },
    transcript::{VerificationError, VerificationResult},
};

impl<M: Embedding + Default> ProtocolConfig<M> {
    /// Derive a canonical commitment sequence before sampling prover randomness.
    pub fn derive(spec: SecuritySpec, tuning: TuningSpec) -> Result<Self, DeriveError> {
        if spec.mode.is_zk() && !matches!(spec.decoding_regime, DecodingRegime::Johnson) {
            return Err(DeriveError::Invalid("Zook requires Johnson decoding"));
        }
        let layout = round_layout::<M>(&spec, &tuning)?;
        let has_rounds = layout.len() > 1;
        let mut targets: Vec<Node<Identity<M::Target>>> = Vec::new();
        for (index, context) in layout
            .iter()
            .enumerate()
            .skip(usize::from(has_rounds))
            .rev()
        {
            let target = targets.last().map(|node| &node.code);
            targets.push(solve_node(
                &spec,
                solve_irs(&spec, context)?,
                target,
                index,
            )?);
        }
        targets.reverse();
        let first = if has_rounds {
            Some(solve_node(
                &spec,
                solve_irs::<M>(&spec, &layout[0])?,
                Some(&targets[0].code),
                0,
            )?)
        } else {
            None
        };
        let mut config = Self {
            request: ProtocolSpec {
                embedding: crate::type_info::Typed::new(M::default()),
                security: spec,
                tuning,
            },
            first,
            targets,
            joint_masks: None,
        };
        if config.security().mode.is_zk() {
            let sumchecks = layout[..layout.len() - 1]
                .iter()
                .map(|context| mask_coefficients(context.folding_factor as usize))
                .collect::<Result<Vec<_>, _>>()?;
            let witnesses = (0..=config.num_rounds())
                .map(|i| config.witness_mask(i).unwrap().num_coefficients())
                .collect::<Result<Vec<_>, _>>()?;
            let degree = if config.has_rounds() {
                embedding_degree::<M>()?
            } else {
                1
            };
            config.joint_masks = Some(solve_joint_masks(
                config.security(),
                &sumchecks,
                &witnesses,
                config.final_commit(),
                degree,
            )?);
        }
        Ok(config)
    }
}

pub(super) fn solve_node<M: Embedding>(
    spec: &SecuritySpec,
    code: IrsConfig<M>,
    target: Option<&IrsConfig<Identity<M::Target>>>,
    index: usize,
) -> Result<Node<M>, DeriveError> {
    if spec.mode.is_zk() {
        WitnessMaskConfig::for_code(&code, usize::from(index != 0))?;
    }
    let (analytic, event, switch) = if let Some(target) = target {
        let ood = if spec.mode.is_zk() {
            WitnessMaskConfig::for_code(target, 1)?;
            1
        } else {
            solve_t_ood(spec, &code, target, index)?
        };
        let switch_pow = grind_to_at(
            spec,
            code_switch_error_bits(&code, target, ood, spec.mode.is_zk())?,
            "code switch",
        )?;
        (
            sumcheck_error_bits(&code, spec.mode.is_zk())?,
            "sumcheck",
            Some((ood, switch_pow)),
        )
    } else {
        let analytic = if spec.mode.is_zk() {
            validate_zook_code(&code, Some(1))?;
            revealed_sumcheck_error_bits::<M::Target>()
        } else {
            sumcheck_error_bits(&code, false)?
        };
        (analytic, "final sumcheck", None)
    };
    Ok(Node {
        code,
        sumcheck_pow: grind_to_at(spec, analytic, event)?,
        switch,
    })
}

fn round_layout<M: Embedding + Default>(
    spec: &SecuritySpec,
    tuning: &TuningSpec,
) -> Result<Vec<RoundContext>, DeriveError> {
    if !tuning.vector_size.is_power_of_two() {
        return Err(DeriveError::Invalid("vector length must be a power of two"));
    }
    if tuning.folding_factor.contains(&0) {
        return Err(DeriveError::Invalid("folding factors must be positive"));
    }
    if tuning.starting_log_inv_rate == 0
        || matches!(
            tuning.rate_schedule,
            RateSchedule::Capped {
                max_log_inv_rate: 0
            }
        )
    {
        return Err(DeriveError::Invalid("inverse rates must be positive"));
    }
    let mut source = RoundContext {
        vector_size: tuning.vector_size,
        log_inv_rate: tuning.starting_log_inv_rate,
        folding_factor: 0,
    };
    if spec.mode.is_zk() {
        while !mask_fits::<Identity<M::Target>>(spec, &source, 0)? {
            source.vector_size = source
                .vector_size
                .checked_mul(2)
                .ok_or(DeriveError::SizeOverflow)?;
        }
    }
    // An oversized initial fold selects the basecase, as in the original layout.
    source.folding_factor = u32::try_from(tuning.folding_factor[0]).unwrap_or(u32::MAX);
    let mut layout = Vec::new();
    while let Some(remaining) =
        (source.vector_size.ilog2() as usize).checked_sub(source.folding_factor as usize)
    {
        let next_folding = tuning.folding_factor[1];
        if !spec.mode.is_zk() && remaining < next_folding {
            break;
        }
        let mut target = RoundContext {
            vector_size: 1usize << remaining,
            log_inv_rate: tuning
                .rate_schedule
                .step(source.log_inv_rate, source.folding_factor),
            folding_factor: u32::try_from(next_folding).map_err(|_| DeriveError::SizeOverflow)?,
        };
        if spec.mode.is_zk() {
            let fits = if layout.is_empty() {
                mask_fits::<M>(spec, &source, 0)?
            } else {
                mask_fits::<Identity<M::Target>>(spec, &source, 1)?
            };
            let final_code = RoundContext {
                folding_factor: 0,
                ..target
            };
            if !fits || !mask_fits::<Identity<M::Target>>(spec, &final_code, 1)? {
                break;
            }
            if remaining < next_folding || !mask_fits::<Identity<M::Target>>(spec, &target, 1)? {
                target.folding_factor = 0;
            }
        }
        layout.push(source);
        source = target;
        if source.folding_factor == 0 {
            break;
        }
    }
    source.folding_factor = 0;
    layout.push(source);
    if let RateSchedule::Adaptive { knee_weight } = tuning.rate_schedule {
        if layout.len() > 1 {
            let rates = plan_adaptive_rates::<M>(spec, tuning, knee_weight, &layout)?;
            for (context, rate) in layout.iter_mut().skip(1).zip(rates) {
                context.log_inv_rate = rate;
            }
        }
    }
    Ok(layout)
}

fn mask_fits<M: Embedding + Default>(
    spec: &SecuritySpec,
    context: &RoundContext,
    incoming: usize,
) -> Result<bool, DeriveError> {
    match WitnessMaskConfig::for_code(&solve_irs::<M>(spec, context)?, incoming) {
        Ok(_) => Ok(true),
        Err(DeriveError::MaskDoesNotFit { .. }) => Ok(false),
        Err(error) => Err(error),
    }
}

const T_OOD_MAX_ITER: usize = 32;

/// Standard WHIR can repeat DEEP samples to reach its analytic target.
pub(super) fn solve_t_ood<M: Embedding>(
    spec: &SecuritySpec,
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
    round_index: usize,
) -> Result<usize, DeriveError> {
    if matches!(spec.decoding_regime, DecodingRegime::Unique) {
        return Ok(1);
    }
    let security_target = f64::from(spec.protocol_security_target_bits());
    let per_sample = M::Target::field_size_bits() - usize_to_f64(source.message_length()).log2();
    let list_bits = target.list_size().log2();
    (1..=T_OOD_MAX_ITER)
        .find(|&t| usize_to_f64(t) * per_sample - 2.0 * list_bits + 1.0 >= security_target)
        .ok_or(DeriveError::FixedPointDidNotConverge { round_index })
}

/// Encode the supplied vector without appending randomness.
pub(super) fn solve_irs<M: Embedding + Default>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
) -> Result<IrsConfig<M>, DeriveError> {
    solve_with_num_vectors(spec, ctx, 1)
}

/// Encode equal-length vectors in one commitment with shared query paths.
pub(super) fn solve_with_num_vectors<M: Embedding + Default>(
    spec: &SecuritySpec,
    ctx: &RoundContext,
    num_vectors: usize,
) -> Result<IrsConfig<M>, DeriveError> {
    let depth = 1_usize
        .checked_shl(ctx.folding_factor)
        .ok_or(DeriveError::SizeOverflow)?;
    if ctx.vector_size == 0 || !ctx.vector_size.is_multiple_of(depth) {
        return Err(DeriveError::InvalidMaskLayout);
    }
    let rate = rate(f64::from(ctx.log_inv_rate));
    if rate == 0.0 {
        return Err(DeriveError::SizeOverflow);
    }
    // Zook query steps sum two error terms; reserve half the error budget
    // for the in-domain term. No grinding credit is used for these IOP bounds.
    let security_target = match spec.mode {
        Mode::Standard => f64::from(spec.protocol_security_target_bits()),
        Mode::ZeroKnowledge => f64::from(spec.target_security_bits) + 1.0,
    };
    Ok(IrsConfig::try_new(IrsParams {
        security_target,
        decoding_regime: spec.decoding_regime,
        hash_id: spec.hash_id,
        num_vectors,
        vector_size: ctx.vector_size,
        interleaving_depth: depth,
        rate,
        mode: IrsMode::Standard,
    })?)
}

impl<M: Embedding + Default> ProtocolConfig<M> {
    pub(crate) fn pair_source(&self) -> VerificationResult<Node<M>> {
        if self.tuning().vector_size != self.padded_vector_size()
            || self.tuning().vector_size.checked_mul(2).is_none()
        {
            return Err(VerificationError);
        }
        let first = self.first_round().ok_or(VerificationError)?;
        if let Some(joint) = self.joint_masks() {
            // Two initial mask-binding events share this error budget.
            if super::security::zook_mask_binding_bits(&joint.commit)
                .map_err(|_| VerificationError)?
                - Bits::new(1.0)
                < Bits::new(f64::from(self.security().target_security_bits))
            {
                return Err(VerificationError);
            }
        }
        // The tuple code retains degree, rate, distance, and list size.
        let source = first
            .code_switch
            .source()
            .joint_pair_source()
            .ok_or(VerificationError)?;
        solve_node(self.security(), source, Some(first.code_switch.target()), 0)
            .map_err(|_| VerificationError)
    }

    pub(crate) fn pair_round<'a>(
        &'a self,
        node: &'a Node<M>,
    ) -> VerificationResult<RoundConfig<'a, M>> {
        let first = self.first_round().ok_or(VerificationError)?;
        Ok(node.round(first.code_switch.target(), 0, self.security().mode.is_zk()))
    }
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum DeriveError {
    #[error("{0}")]
    Invalid(&'static str),
    #[error("protocol dimensions overflow usize")]
    SizeOverflow,
    #[error("mask requires {required} coefficients, polynomial has {available}")]
    MaskDoesNotFit { required: usize, available: usize },
    #[error("invalid joint-mask layout")]
    InvalidMaskLayout,
    #[error("t_ood fixed-point did not converge for round {round_index}")]
    FixedPointDidNotConverge { round_index: usize },
    #[error("{pow} cannot be ground: {source}")]
    PowUngrindable {
        pow: &'static str,
        #[source]
        source: PowError,
    },
    #[error("{pow} requires {required} bits, exceeds pow_budget = {max}")]
    PowBudgetExceeded {
        pow: &'static str,
        required: Bits,
        max: Bits,
    },
    #[error("codeword length {length} exceeds the NTT engine's supported order")]
    CodewordExceedsNtt { length: usize },
    #[error("{pow} soundness gap: analytic {analytic} < target {target}")]
    SecurityTargetNotMet {
        pow: &'static str,
        analytic: Bits,
        target: Bits,
    },
    #[error("adaptive rate planner found no feasible schedule")]
    AdaptiveNoFeasibleSchedule,
}

impl From<CodewordLengthError> for DeriveError {
    fn from(error: CodewordLengthError) -> Self {
        Self::CodewordExceedsNtt {
            length: error.length,
        }
    }
}
