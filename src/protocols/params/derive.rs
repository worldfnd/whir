//! Construction and execution use the same bounds and mask dimensions.
use ark_ff::Field;
use security::{grind_to_at, rate, round_bits, usize_to_f64};
use thiserror::Error;

use super::{
    embedding_degree, masks, require, search, security, DecodingRegime, Params, ProtocolConfig,
    ProtocolSpec, RateSchedule, Result, RoundConfig, RoundContext, WitnessMaskConfig,
};
use crate::{
    algebra::{
        embedding::{Embedding, Identity},
        fields::FieldWithSize,
    },
    bits::Bits,
    protocols::{
        code_switch::{CodeSwitchMode, Config as CodeSwitchConfig},
        irs_commit::{Config as IrsConfig, IrsMode, IrsParams},
        sumcheck::{Config as SumcheckConfig, SumcheckMaskLen, SumcheckMode},
    },
    transcript::{VerificationError, VerificationResult},
    type_info::Typed,
};

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{0}")]
pub struct DeriveError(pub &'static str);

impl<M: Embedding + Default> ProtocolConfig<M> {
    pub fn derive(spec: Params) -> Result<Self> {
        require(
            !spec.zk || spec.regime == DecodingRegime::Johnson,
            "Zook requires Johnson decoding",
        )?;
        let layout = round_layout::<M>(&spec)?;
        Self::build(&spec, &layout)
    }
    pub(super) fn build(spec: &Params, layout: &[RoundContext]) -> Result<Self> {
        let has_rounds = layout.len() > 1;
        let targets = layout
            .iter()
            .skip(usize::from(has_rounds))
            .map(|ctx| solve_irs(spec, ctx))
            .collect::<Result<Vec<_>>>()?;
        let first = has_rounds
            .then(|| solve_irs::<M>(spec, &layout[0]))
            .transpose()?;
        let mut config = Self {
            request: ProtocolSpec {
                embedding: Typed::new(M::default()),
                params: *spec,
            },
            first,
            targets,
            joint_masks: None,
        };
        if let Some(first) = &config.first {
            round(spec, first, &config.targets[0], 0)?;
        }
        for (index, codes) in config.targets.windows(2).enumerate() {
            round(spec, &codes[0], &codes[1], index + 1)?;
        }
        let _ = terminal(spec, config.final_commit(), usize::from(has_rounds))?;
        if spec.zk {
            let sumchecks: Vec<_> = layout[..layout.len() - 1]
                .iter()
                .map(|ctx| 2 * ctx.folds as usize + 1)
                .collect();
            let witnesses = (0..=config.num_rounds())
                .map(|i| config.witness_mask(i).unwrap().num_coefficients())
                .collect::<Result<Vec<_>>>()?;
            let degree = if has_rounds {
                embedding_degree::<M>()?
            } else {
                1
            };
            config.joint_masks = Some(masks::joint_code(
                spec,
                &sumchecks,
                &witnesses,
                config.final_commit(),
                degree,
            )?);
        }
        Ok(config)
    }
    pub(crate) fn pair_source(&self) -> VerificationResult<IrsConfig<M>> {
        if self.params().size != self.padded_vector_size()
            || self.params().size.checked_mul(2).is_none()
        {
            return Err(VerificationError);
        }
        let first = self.first.as_ref().ok_or(VerificationError)?;
        if let Some(joint) = self.joint_masks() {
            if security::final_bits(self.final_commit(), &joint.commit)
                .map(|bits| bits.0)
                .map_err(|_| VerificationError)?
                - Bits::new(1.0)
                < Bits::new(f64::from(self.params().bits))
            {
                return Err(VerificationError);
            }
        }
        first.joint_pair_source().ok_or(VerificationError)
    }
    pub(crate) fn pair_round<'a>(
        &'a self,
        source: &'a IrsConfig<M>,
    ) -> VerificationResult<RoundConfig<'a, M>> {
        self.first.as_ref().ok_or(VerificationError)?;
        round(self.params(), source, &self.targets[0], 0).map_err(|_| VerificationError)
    }
}

pub(super) fn round<'a, M: Embedding>(
    spec: &Params,
    source: &'a IrsConfig<M>,
    target: &'a IrsConfig<Identity<M::Target>>,
    index: usize,
) -> Result<RoundConfig<'a, M>> {
    let zk = spec.zk;
    let masks = zk
        .then(|| WitnessMaskConfig::for_code(source, usize::from(index != 0)))
        .transpose()?;
    if zk {
        WitnessMaskConfig::for_code(target, 1)?;
    }
    let ood = if zk {
        1
    } else {
        solve_t_ood(spec, source, target)?
    };
    let (sumcheck, switch) = round_bits(source, Some(target), ood, zk)?;
    let mode = if zk {
        SumcheckMode::ZeroKnowledge {
            mask_length: SumcheckMaskLen::new(3),
        }
    } else {
        SumcheckMode::Standard
    };
    Ok(RoundConfig {
        index,
        masks,
        sumcheck: SumcheckConfig::new(
            source.vector_size(),
            grind_to_at(spec, sumcheck, "sumcheck")?,
            source.interleaving_depth().ilog2() as usize,
            mode,
        ),
        code_switch: CodeSwitchConfig::new(
            source,
            target,
            ood,
            CodeSwitchMode::Standard,
            grind_to_at(spec, switch.unwrap(), "code switch")?,
        ),
    })
}
pub(super) fn terminal<F: Field>(
    spec: &Params,
    code: &IrsConfig<Identity<F>>,
    incoming: usize,
) -> Result<SumcheckConfig<F>> {
    if spec.zk {
        WitnessMaskConfig::for_code(code, incoming)?;
    }
    let (bits, _) = round_bits(code, None, 1, spec.zk)?;
    Ok(SumcheckConfig::new(
        code.vector_size(),
        grind_to_at(spec, bits, "final sumcheck")?,
        code.vector_size().ilog2() as usize,
        SumcheckMode::Standard,
    ))
}

fn round_layout<M: Embedding + Default>(spec: &Params) -> Result<Vec<RoundContext>> {
    if !spec.size.is_power_of_two() {
        return Err(DeriveError("vector length must be a power of two"));
    }
    if spec.folds.contains(&0) {
        return Err(DeriveError("folding factors must be positive"));
    }
    if spec.rate == 0 || matches!(spec.schedule, RateSchedule::Capped(0)) {
        return Err(DeriveError("inverse rates must be positive"));
    }
    let mut source = RoundContext {
        size: spec.size,
        rate: spec.rate,
        folds: 0,
    };
    if spec.zk {
        while !mask_fits::<Identity<M::Target>>(spec, &source, 0)? {
            source.size = source
                .size
                .checked_mul(2)
                .ok_or(DeriveError("dimension overflow"))?;
        }
    }
    // An oversized initial fold selects the basecase, as in the original layout.
    source.folds = u32::try_from(spec.folds[0]).unwrap_or(u32::MAX);
    let mut layout = Vec::new();
    while let Some(remaining) = (source.size.ilog2() as usize).checked_sub(source.folds as usize) {
        let next_folding = spec.folds[1];
        if !spec.zk && remaining < next_folding {
            break;
        }
        let mut target = RoundContext {
            size: 1usize << remaining,
            rate: spec.schedule.step(source.rate, source.folds),
            folds: u32::try_from(next_folding).map_err(|_| DeriveError("dimension overflow"))?,
        };
        if spec.zk {
            let fits = if layout.is_empty() {
                mask_fits::<M>(spec, &source, 0)?
            } else {
                mask_fits::<Identity<M::Target>>(spec, &source, 1)?
            };
            let final_code = RoundContext { folds: 0, ..target };
            if !fits || !mask_fits::<Identity<M::Target>>(spec, &final_code, 1)? {
                break;
            }
            if remaining < next_folding || !mask_fits::<Identity<M::Target>>(spec, &target, 1)? {
                target.folds = 0;
            }
        }
        layout.push(source);
        source = target;
        if source.folds == 0 {
            break;
        }
    }
    source.folds = 0;
    layout.push(source);
    if let RateSchedule::Adaptive(knee_weight) = spec.schedule {
        if layout.len() > 1 {
            let rates =
                search::plan_adaptive_rates::<M>(spec, knee_weight.0.into_inner(), &layout)?;
            for (context, rate) in layout.iter_mut().skip(1).zip(rates) {
                context.rate = rate;
            }
        }
    }
    Ok(layout)
}

fn mask_fits<M: Embedding + Default>(
    spec: &Params,
    context: &RoundContext,
    incoming: usize,
) -> Result<bool> {
    match WitnessMaskConfig::for_code(&solve_irs::<M>(spec, context)?, incoming) {
        Ok(_) => Ok(true),
        Err(DeriveError("mask does not fit")) => Ok(false),
        Err(error) => Err(error),
    }
}

/// Standard WHIR can repeat DEEP samples to reach its analytic target.
pub(super) fn solve_t_ood<M: Embedding>(
    spec: &Params,
    source: &IrsConfig<M>,
    target: &IrsConfig<Identity<M::Target>>,
) -> Result<usize> {
    if matches!(spec.regime, DecodingRegime::Unique) {
        return Ok(1);
    }
    let security_target = f64::from(spec.bits.saturating_sub(spec.pow));
    let per_sample = M::Target::field_size_bits() - usize_to_f64(source.message_length()).log2();
    let list_bits = target.list_size().log2();
    (1..=32)
        .find(|&t| usize_to_f64(t) * per_sample - 2.0 * list_bits + 1.0 >= security_target)
        .ok_or(DeriveError("DEEP sample bound exceeds 32"))
}

/// Encode the supplied vector without appending randomness.
pub(super) fn solve_irs<M: Embedding + Default>(
    spec: &Params,
    ctx: &RoundContext,
) -> Result<IrsConfig<M>> {
    solve_with_num_vectors(spec, ctx, 1)
}

/// Encode equal-length vectors in one commitment with shared query paths.
pub(super) fn solve_with_num_vectors<M: Embedding + Default>(
    spec: &Params,
    ctx: &RoundContext,
    num_vectors: usize,
) -> Result<IrsConfig<M>> {
    let depth = 1_usize
        .checked_shl(ctx.folds)
        .ok_or(DeriveError("dimension overflow"))?;
    if ctx.size == 0 || !ctx.size.is_multiple_of(depth) {
        return Err(DeriveError("invalid code dimensions"));
    }
    let rate = rate(f64::from(ctx.rate));
    if rate == 0.0 {
        return Err(DeriveError("dimension overflow"));
    }
    // Zook query steps sum two error terms; reserve half the error budget
    // for the in-domain term. No grinding credit is used for these IOP bounds.
    let security_target = if spec.zk {
        f64::from(spec.bits) + 1.0
    } else {
        f64::from(spec.bits.saturating_sub(spec.pow))
    };
    IrsConfig::try_new(IrsParams {
        security_target,
        decoding_regime: spec.regime,
        hash_id: spec.hash,
        num_vectors,
        vector_size: ctx.size,
        interleaving_depth: depth,
        rate,
        mode: IrsMode::Standard,
    })
    .map_err(|_| DeriveError("codeword exceeds NTT order"))
}
