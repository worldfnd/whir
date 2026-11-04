use ark_ff::{AdditiveGroup, FftField, Field};

use super::{oracle::Openings, Config, SwitchOpenings, TensorLinearForm};
use crate::{
    algebra::{
        dot,
        fields::{Field128, Field256, Field64},
        ntt::generator,
        univariate_evaluate,
    },
    hash::{BLAKE3, COPY, SHA2},
    protocols::matrix_commit::Encodable,
    transcript::{
        codecs::Empty, Codec, DomainSeparator, Proof, ProverState, VerificationResult,
        VerifierMessage, VerifierState,
    },
};

fn prove(
    size: usize,
    folds: usize,
    ood: usize,
) -> (
    Config<Field256>,
    TensorLinearForm<Field256>,
    Field256,
    Proof,
) {
    let config = Config::new(size, folds, 8, ood, 2, 64, SHA2).unwrap();
    let point: Vec<_> = (0..size.next_power_of_two().ilog2())
        .map(|i| Field256::from(u64::from(i) + 3))
        .collect();
    let form = TensorLinearForm::multilinear(&point);
    let vector: Vec<_> = (0..size)
        .map(|i| Field256::from((i * i + 3) as u64))
        .collect();
    let value = dot(&form.coefficients()[..size], &vector);
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    let mut state = ProverState::new_std(&ds);
    let witness = config.commit(&mut state, &vector).unwrap();
    config.prove(&mut state, witness, &form, value).unwrap();
    (config, form, value, state.proof())
}

fn verify(
    config: &Config<Field256>,
    form: &TensorLinearForm<Field256>,
    value: Field256,
    proof: &Proof,
) -> VerificationResult<()> {
    let ds = DomainSeparator::protocol(config).instance(&Empty);
    let mut state = VerifierState::new_std(&ds, proof);
    let commitment = config.receive_commitment(&mut state)?;
    config.verify(&mut state, commitment, form, value)?;
    state.check_eof()
}

#[test]
fn proves_base_and_multiple_code_switches_with_multiple_oods() {
    for folds in [3, 4, 5] {
        for size in [1, 8, 9, 255, 256, 257, 4096] {
            for ood in [0, 1, 3] {
                let (config, form, value, proof) = prove(size, folds, ood);
                verify(&config, &form, value, &proof).unwrap();
            }
        }
    }
}

#[test]
#[cfg(not(feature = "verifier_panics"))]
fn rejects_changed_claim_and_form() {
    let (config, form, value, proof) = prove(256, 3, 2);
    assert!(verify(&config, &form, value + Field256::ONE, &proof).is_err());
    let other = TensorLinearForm::multilinear(&[Field256::from(5); 8]);
    assert!(verify(&config, &other, value, &proof).is_err());
}

#[test]
#[cfg(not(feature = "verifier_panics"))]
fn enforces_every_initial_ood_claim() {
    let (config, form, value, proof) = prove(256, 3, 2);
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    for index in 0..2 {
        let mut state = VerifierState::new_std(&ds, &proof);
        let mut commitment = config.receive_commitment(&mut state).unwrap();
        commitment.ood[index].1 += Field256::ONE;
        assert!(config.verify(&mut state, commitment, &form, value).is_err());
    }
}

#[test]
fn proves_zero_form_at_base_and_after_code_switches() {
    for size in [8, 256] {
        let config = Config::<Field256>::new(size, 3, 8, 0, 2, 64, SHA2).unwrap();
        let vector: Vec<_> = (0..size).map(|i| Field256::from(i as u64)).collect();
        let form = TensorLinearForm::zero(size);
        let ds = DomainSeparator::protocol(&config).instance(&Empty);
        let mut state = ProverState::new_std(&ds);
        let witness = config.commit(&mut state, &vector).unwrap();
        config
            .prove(&mut state, witness, &form, Field256::ZERO)
            .unwrap();
        verify(&config, &form, Field256::ZERO, &state.proof()).unwrap();
    }
}

#[test]
#[cfg(not(feature = "verifier_panics"))]
fn rejects_corrupted_rows() {
    let (config, form, value, proof) = prove(256, 3, 2);
    for index in [0, proof.hints.len() / 2, proof.hints.len() - 1] {
        let mut corrupted = proof.clone();
        corrupted.hints[index] ^= 1;
        assert!(verify(&config, &form, value, &corrupted).is_err());
    }
}

// v1's debug-only interaction trace asserts on changed Fiat–Shamir paths.
// Exercise arbitrary transcript corruption against the production verifier.
#[test]
#[cfg(all(not(debug_assertions), not(feature = "verifier_panics")))]
fn rejects_corrupted_transcript_fields() {
    let (config, form, value, proof) = prove(256, 3, 2);
    for index in [0, proof.narg_string.len() / 2, proof.narg_string.len() - 1] {
        let mut corrupted = proof.clone();
        corrupted.narg_string[index] ^= 1;
        assert!(verify(&config, &form, value, &corrupted).is_err());
    }
}

#[test]
#[cfg(all(not(debug_assertions), not(feature = "verifier_panics")))]
fn rejects_modified_terminal_table() {
    let (config, form, value, proof) = prove(256, 3, 2);
    let size = config.dimensions().last().unwrap().table_size();
    let tail_fields = size + 2 * size.ilog2() as usize;
    let start = proof.narg_string.len() - tail_fields * Field256::encoded_size();
    for index in [0, size / 2, size - 1] {
        let mut corrupted = proof.clone();
        corrupted.narg_string[start + index * Field256::encoded_size()] ^= 1;
        assert!(verify(&config, &form, value, &corrupted).is_err());
    }
}

#[test]
fn rejects_invalid_dimensions_and_nonshrinking_schedules() {
    assert!(Config::<Field256>::new(256, 3, 8, 1, 2, 64, COPY).is_err());
    assert!(Config::<Field256>::new(256, 3, 8, 1, 2, 64, [0; 32].into()).is_err());
    assert!(Config::<Field256>::new(256, 2, 8, 1, 2, 64, SHA2).is_err());
    assert!(Config::<Field256>::new(0, 2, 8, 1, 2, 64, SHA2).is_err());
    assert!(Config::<Field256>::new(256, 0, 8, 1, 2, 64, SHA2).is_err());
    assert!(Config::<Field256>::new(256, 2, 0, 1, 2, 64, SHA2).is_err());
    assert!(Config::<Field256>::new(256, 2, 8, 1, 1, 64, SHA2).is_err());
    assert!(Config::<Field256>::new(256, 2, 64, 1, 2, 64, SHA2).is_err());
}

#[test]
fn prefix_masks_fit_in_complete_age_check_padding() {
    let capacity = 1 << 20;
    let prefix_capacity = capacity - 32 * 256;
    for size in [
        657_900 + 80,
        655_177,
        prefix_capacity,
        prefix_capacity + 1,
        capacity,
    ] {
        let config = Config::<Field256>::new(size, 5, 148, 1, 2, 2048, SHA2).unwrap();
        let dimensions = config.dimensions()[0];
        assert_eq!(dimensions.prefix, 256);
        assert_eq!(
            dimensions.table_size(),
            if size <= prefix_capacity {
                capacity
            } else {
                2 * capacity
            }
        );
    }
}

#[test]
fn unsalted_rows_work_across_field_encodings() {
    fn check<F: FftField + Codec>() {
        let config = Config::<F>::new(256, 3, 8, 1, 2, 64, BLAKE3).unwrap();
        for &dimensions in config.dimensions() {
            let width = dimensions.row_width::<F>();
            let block_fields = 64 / F::encoded_size();
            assert_eq!(
                width,
                (dimensions.columns + 1).next_multiple_of(block_fields)
            );
        }
        let vector: Vec<_> = (0..256).map(F::from).collect();
        let form = TensorLinearForm::multilinear(&[F::from(3); 8]);
        let value = dot(&form.coefficients(), &vector);
        let ds = DomainSeparator::protocol(&config).instance(&Empty);
        let mut state = ProverState::new_std(&ds);
        let witness = config.commit(&mut state, &vector).unwrap();
        config.prove(&mut state, witness, &form, value).unwrap();
        let proof = state.proof();
        let mut state = VerifierState::new_std(&ds, &proof);
        let commitment = config.receive_commitment(&mut state).unwrap();
        config.verify(&mut state, commitment, &form, value).unwrap();
        state.check_eof().unwrap();
    }
    check::<Field64>();
    check::<Field128>();
    check::<Field256>();
}

#[test]
fn proves_with_fft_batched_final_openings() {
    let config = Config::<Field256>::new(4096, 3, 128, 1, 2, 1024, SHA2).unwrap();
    let dimensions = config.dimensions().last().unwrap();
    assert!(128 * dimensions.table_size() > 2 * dimensions.rows * dimensions.rows.ilog2() as usize);
    let vector: Vec<_> = (0..4096).map(Field256::from).collect();
    let form = TensorLinearForm::multilinear(&[Field256::from(3); 12]);
    let value = dot(&form.coefficients(), &vector);
    let ds = DomainSeparator::protocol(&config).instance(&Empty);
    let mut state = ProverState::new_std(&ds);
    let witness = config.commit(&mut state, &vector).unwrap();
    config.prove(&mut state, witness, &form, value).unwrap();
    verify(&config, &form, value, &state.proof()).unwrap();
}

#[test]
fn compressed_mask_claims_match_univariate_constraints() {
    for (queries, base) in [(64, 512), (343, 2048)] {
        let config = Config::<Field256>::new(4096, 3, queries, 1, 2, base, SHA2).unwrap();
        let ds = DomainSeparator::protocol(&config).instance(&Empty);
        let mut state = ProverState::new_std(&ds);
        let mut reference = ProverState::new_std(&ds);
        let mask = config.mask_dimensions();
        let root = generator::<Field256>(mask.rows).unwrap();
        let indices = if queries == 343 {
            let mut indices: Vec<_> = (0..queries).collect();
            indices[2] = mask.rows - 1;
            indices[queries - 1] = 1;
            indices
        } else {
            vec![0, 1, mask.rows - 1, 1]
        };
        let points = indices
            .iter()
            .map(|&index| root.pow([index as u64]))
            .collect::<Vec<_>>();
        let mut mask_table = vec![Field256::ZERO; mask.column_size];
        for (index, entry) in mask_table[..mask.prefix + 7].iter_mut().enumerate() {
            *entry = Field256::from(index as u64 + 3);
        }
        let values = points
            .iter()
            .map(|&point| vec![univariate_evaluate(&mask_table, point)])
            .collect();
        let mut openings = SwitchOpenings {
            ood: Vec::new(),
            main: Openings {
                indices: Vec::new(),
                points: Vec::new(),
                values: Vec::new(),
            },
            mask: Openings {
                indices,
                points,
                values,
            },
        };
        let folded_size = config.dimensions()[0].column_size;
        let next = config.dimensions()[1];
        let batch = config.batch_switch(&mut state, 0, &[Field256::from(5); 3], &openings);
        let mut payload = vec![Field256::ZERO; next.payload_size];
        payload[folded_size..folded_size + 7]
            .copy_from_slice(&mask_table[mask.prefix..mask.prefix + 7]);
        let prefix_start = folded_size + config.mask_segment_size();
        payload[prefix_start..prefix_start + mask.prefix]
            .copy_from_slice(&mask_table[..mask.prefix]);
        let coefficients = batch.payload.coefficients();
        assert_eq!(dot(&coefficients, &payload), batch.sum);
        for mut range in [
            folded_size..folded_size + 7,
            prefix_start..prefix_start + mask.prefix,
        ] {
            let index = range
                .find(|&index| coefficients[index] != Field256::ZERO)
                .unwrap();
            payload[index] += Field256::ONE;
            assert_ne!(dot(&coefficients, &payload), batch.sum);
            payload[index] -= Field256::ONE;
        }
        let mut expected = TensorLinearForm::zero(next.payload_size);
        let mut sum = Field256::ZERO;
        for (&point, row) in openings.mask.points.iter().zip(&openings.mask.values) {
            let scalar: Field256 = reference.verifier_message();
            expected.add_univariate(
                folded_size,
                7,
                point,
                scalar * point.pow([mask.prefix as u64]),
            );
            expected.add_univariate(prefix_start, mask.prefix, point, scalar);
            sum += scalar * row[0];
        }
        assert!(batch
            .table
            .coefficients()
            .iter()
            .all(|&weight| weight == Field256::ZERO));
        assert_eq!(coefficients, expected.coefficients());
        assert_eq!(batch.sum, sum);
        for (&point, row) in openings.mask.points.iter().zip(&mut openings.mask.values) {
            row[0] += point.pow([(mask.prefix + 7) as u64]);
        }
        let mut malformed_state = ProverState::new_std(&ds);
        let malformed =
            config.batch_switch(&mut malformed_state, 0, &[Field256::from(5); 3], &openings);
        assert_ne!(
            dot(&malformed.payload.coefficients(), &payload),
            malformed.sum
        );
    }
}
