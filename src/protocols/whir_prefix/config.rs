use ark_ff::FftField;
use serde::Serialize;

use crate::{
    algebra::{
        fields::FieldWithSize,
        ntt::{generator, NTT},
    },
    engines::EngineId,
    hash::ENGINES,
    protocols::{
        matrix_commit::Encodable, proof_of_work::Config as PowConfig,
        sumcheck::Config as SumcheckConfig,
    },
    type_info::Type,
};

/// Explicit experimental parameters; query counts are not certified security levels.
#[derive(Clone, Debug, Serialize)]
#[serde(bound = "")]
pub struct Config<F: FftField> {
    protocol_version: &'static str,
    pub(crate) field: Type<F>,
    pub(crate) witness_size: usize,
    pub(crate) folds: usize,
    pub(crate) queries: usize,
    pub(crate) ood_samples: usize,
    pub(crate) log_inv_rate: usize,
    pub(crate) hash_id: EngineId,
    pub(crate) rounds: Vec<Dimensions>,
    pub(crate) final_sumcheck: SumcheckConfig<F>,
    mask: Dimensions,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Dimensions {
    pub payload_size: usize,
    pub columns: usize,
    pub prefix: usize,
    pub column_size: usize,
    pub rows: usize,
}

impl Dimensions {
    fn new<F: FftField>(
        payload_size: usize,
        columns: usize,
        prefix: usize,
        log_inv_rate: usize,
        hash_id: EngineId,
    ) -> Result<Self, &'static str> {
        if NTT.get::<F>().is_none() {
            return Err("unregistered Reed-Solomon encoder");
        }
        let column_size = payload_size
            .div_ceil(columns)
            .checked_add(prefix)
            .and_then(usize::checked_next_power_of_two)
            .ok_or("column size overflow")?;
        let rows = column_size
            .checked_mul(1usize << log_inv_rate)
            .ok_or("codeword size overflow")?;
        let sampling_size = rows.checked_add(prefix).ok_or("sampling size overflow")?;
        if generator::<F>(rows).is_none()
            || (sampling_size as f64).log2() >= F::field_size_bits() - 1.0
        {
            return Err("field does not support the requested domain or OOD count");
        }
        let dimensions = Self {
            payload_size,
            columns,
            prefix,
            column_size,
            rows,
        };
        let width = dimensions.row_width::<F>();
        if columns.checked_mul(column_size).is_none() || rows.checked_mul(width).is_none() {
            return Err("oracle allocation size overflow");
        }
        let hash = ENGINES.retrieve(hash_id).ok_or("unknown hash engine")?;
        if !hash.supports_size(64) || !hash.supports_size(width * F::encoded_size()) {
            return Err("hash engine cannot hash Merkle nodes or rows");
        }
        Ok(dimensions)
    }

    pub(super) fn embed_payload<F: FftField>(self, payload: &[F]) -> Vec<F> {
        assert!(payload.len() <= self.payload_size);
        let mut table = vec![F::ZERO; self.table_size()];
        for (j, column) in table.chunks_exact_mut(self.column_size).enumerate() {
            let start = j * self.payload_block();
            let end = (start + self.payload_block()).min(payload.len());
            if start < end {
                column[self.prefix..self.prefix + end - start]
                    .copy_from_slice(&payload[start..end]);
            }
        }
        table
    }

    pub fn row_width<F: FftField>(self) -> usize {
        // Include the oracle identifier and align for block-based hash engines.
        let bytes = F::encoded_size();
        let mut width = self.columns + 1;
        while !(width * bytes).is_multiple_of(64) {
            width += 1;
        }
        width
    }
    pub(crate) const fn table_size(self) -> usize {
        self.columns * self.column_size
    }

    pub(crate) const fn payload_block(self) -> usize {
        self.payload_size.div_ceil(self.columns)
    }
}

impl<F: FftField> Config<F> {
    /// Accepts an unpadded witness; opening forms have length `witness_size.next_power_of_two()`.
    /// Form coefficients beyond `witness_size` are ignored.
    pub fn new(
        witness_size: usize,
        folds: usize,
        queries: usize,
        ood_samples: usize,
        log_inv_rate: usize,
        base_size: usize,
        hash_id: EngineId,
    ) -> Result<Self, &'static str> {
        if witness_size == 0
            || witness_size.checked_next_power_of_two().is_none()
            || !base_size.is_power_of_two()
        {
            return Err(
                "witness size must be positive and fit a power of two; base must be a power of two",
            );
        }
        if !(1..=8).contains(&folds) || queries == 0 {
            return Err("folds must be in 1..=8 and queries must be positive");
        }
        if !(2..usize::BITS as usize).contains(&log_inv_rate) {
            return Err("inverse rate must be a power of two at least four");
        }
        let prefix = queries
            .checked_add(ood_samples)
            .and_then(usize::checked_next_power_of_two)
            .ok_or("prefix overflow")?;
        let mask_prefix = queries
            .checked_next_power_of_two()
            .ok_or("mask prefix overflow")?;
        let mask_payload = (2 * folds + 1).next_power_of_two();
        let mask_segment = mask_payload.max(mask_prefix);
        let minimum_base = prefix
            .checked_add(mask_segment)
            .ok_or("prefix overflow")?
            .checked_add(mask_prefix)
            .and_then(usize::checked_next_power_of_two)
            .ok_or("base size overflow")?;
        if base_size < minimum_base {
            return Err("base size is too small for the carried mask and prefix");
        }
        let mask = Dimensions::new::<F>(mask_payload, 1, mask_prefix, log_inv_rate, hash_id)?;
        let columns = 1usize << folds;
        let mut payload_size = witness_size;
        let mut rounds = Vec::new();
        loop {
            let count = if payload_size <= base_size {
                1
            } else {
                columns
            };
            let dimensions =
                Dimensions::new::<F>(payload_size, count, prefix, log_inv_rate, hash_id)?;
            if rounds.last().is_some_and(|previous: &Dimensions| {
                dimensions.table_size() >= previous.table_size()
            }) {
                return Err("packing does not shrink the table; increase the base size");
            }
            rounds.push(dimensions);
            if count == 1 {
                break;
            }
            payload_size = dimensions
                .column_size
                .checked_add(mask_segment)
                .and_then(|n| n.checked_add(mask_prefix))
                .and_then(usize::checked_next_power_of_two)
                .ok_or("packed size overflow")?;
        }
        let final_size = rounds.last().expect("nonempty schedule").table_size();
        let final_sumcheck = SumcheckConfig {
            field: Type::new(),
            initial_size: final_size,
            round_pow: PowConfig {
                hash_id,
                threshold: u64::MAX,
            },
            num_rounds: final_size.ilog2() as usize,
        };
        Ok(Self {
            protocol_version: "whir-prefix-zk-experiment/v2-unsalted",
            field: Type::new(),
            witness_size,
            folds,
            queries,
            ood_samples,
            log_inv_rate,
            hash_id,
            rounds,
            final_sumcheck,
            mask,
        })
    }

    pub fn dimensions(&self) -> &[Dimensions] {
        &self.rounds
    }

    pub const fn mask_dimensions(&self) -> Dimensions {
        self.mask
    }

    pub(crate) const fn mask_segment_size(&self) -> usize {
        // Pad the coefficient segment so its following prefix is binary-aligned.
        if self.mask.payload_size > self.mask.prefix {
            self.mask.payload_size
        } else {
            self.mask.prefix
        }
    }
}
