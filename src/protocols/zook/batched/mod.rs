//! Multi-polynomial batching for Zook.
//!
//! Two orthogonal batching axes, then the standard round walk:
//!   - **Intra-bundle** ([`bundle`]): `N` same-size polys committed together,
//!     collapsed into one virtual block by a per-bundle γ-RLC.
//!   - **Inter-bundle** ([`selector`]): several separately-committed bundles
//!     merged at matching round lengths by the selector sumcheck.
//!
//! [`prover`] and [`verifier`] each own one side of the protocol end-to-end:
//! intro each bundle into a block, run any per-bundle pre-merge rounds, then
//! walk the shared rounds — draining active blocks through the selector merge
//! and the shared round atom ([`super::round`]) — into the inner basecase.
//!
//! The batched verifier returns the same [`super::FinalClaim`] as the
//! single-track protocol (one [`super::ClaimGroup`] per bundle); the caller
//! finishes via [`super::FinalClaim::verify_bundles`].

pub mod bundle;
pub mod commit;
pub mod prover;
pub mod selector;
pub mod verifier;

#[cfg(test)]
mod tests {
    //! End-to-end roundtrip tests for the batched zook orchestrator (Path A).
    //!
    //! Each test:
    //! 1. Derives a [`BatchedProtocolConfig`] for the requested bundle shapes.
    //! 2. Builds random polys + per-poly multilinear forms with their true
    //!    evaluations as bundle claims.
    //! 3. Runs `commit_bundle` for each bundle on the prover side, then
    //!    `BatchedProtocolConfig::prove`.
    //! 4. Mirrors on the verifier side: `receive_bundle_commitment` per bundle,
    //!    then `verify`, then `FinalClaim::verify_bundles` with the descriptors.
    //!
    //! The tests cover the spec §17 happy paths that fall within the first-pass
    //! scope (same `num_polys` across bundles, no basecase joins).

    use ark_ff::Field as _;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use crate::{
        algebra::{
            fields::Field64,
            linear_form::{Evaluate, MultilinearExtension},
            random_vector,
        },
        hash,
        protocols::{
            params::{
                batched::{BatchedProtocolConfig, BatchedTuningSpec, BundleSpec},
                spec::{
                    DecodingRegime, FoldingFactor, Mode, PowBudget, RateSchedule, SecuritySpec,
                },
                test_utils::TestEmbedding,
                KneeWeight,
            },
            zook::batched::bundle::{BundleClaim, BundleDescriptor, WitnessBundle},
        },
        transcript::{codecs::Empty, DomainSeparator, ProverState, VerifierState},
    };

    type F = Field64;
    type Emb = TestEmbedding;

    fn test_spec(mode: Mode) -> SecuritySpec {
        SecuritySpec {
            mode,
            decoding_regime: DecodingRegime::Johnson,
            target_security_bits: 40,
            pow_budget: PowBudget::per_slot(20),
            hash_id: hash::BLAKE3,
        }
    }

    fn batched_tuning(bundles: Vec<BundleSpec>) -> BatchedTuningSpec {
        BatchedTuningSpec {
            bundles,
            starting_log_inv_rate: 1,
            folding_factor: FoldingFactor::Constant(2),
            rate_schedule: RateSchedule::Adaptive {
                knee_weight: KneeWeight::DEFAULT,
            },
        }
    }

    /// Build a random bundle of `num_polys` polys of length `1 << log_m`, each
    /// carrying `claims_per_poly` random multilinear-extension claims at random
    /// evaluation points (the claim value is the honest poly evaluation).
    struct RandomBundle {
        polys: Vec<Vec<F>>,
        per_poly_forms: Vec<Vec<MultilinearExtension<F>>>,
        per_poly_values: Vec<Vec<F>>,
    }

    fn random_bundle(
        num_polys: usize,
        log_m: usize,
        claims_per_poly: usize,
        seed: u64,
    ) -> RandomBundle {
        let embedding = Emb::default();
        let mut rng = StdRng::seed_from_u64(seed);
        let m = 1usize << log_m;
        let polys: Vec<Vec<F>> = (0..num_polys)
            .map(|_| random_vector::<F>(&mut rng, m))
            .collect();
        let mut per_poly_forms = Vec::with_capacity(num_polys);
        let mut per_poly_values = Vec::with_capacity(num_polys);
        for poly in &polys {
            let mut forms = Vec::with_capacity(claims_per_poly);
            let mut values = Vec::with_capacity(claims_per_poly);
            for _ in 0..claims_per_poly {
                let point: Vec<F> = random_vector(&mut rng, log_m);
                let form = MultilinearExtension { point };
                let value = form.evaluate(&embedding, poly);
                forms.push(form);
                values.push(value);
            }
            per_poly_forms.push(forms);
            per_poly_values.push(values);
        }
        RandomBundle {
            polys,
            per_poly_forms,
            per_poly_values,
        }
    }

    /// Borrowed [`WitnessBundle`] over a [`RandomBundle`] (prover side).
    fn witness_bundle<'a>(rb: &'a RandomBundle) -> WitnessBundle<'a, F> {
        let polys: Vec<&[F]> = rb.polys.iter().map(Vec::as_slice).collect();
        let per_poly_claims: Vec<Vec<BundleClaim<'a, F>>> = rb
            .per_poly_forms
            .iter()
            .zip(&rb.per_poly_values)
            .map(|(forms, values)| {
                forms
                    .iter()
                    .zip(values)
                    .map(|(form, &value)| BundleClaim { form, value })
                    .collect()
            })
            .collect();
        WitnessBundle {
            polys,
            per_poly_claims,
        }
    }

    /// Verifier-side [`BundleDescriptor`] for a [`RandomBundle`].
    fn bundle_descriptor<'a>(
        rb: &'a RandomBundle,
        num_polys: usize,
        length: usize,
    ) -> BundleDescriptor<'a, F> {
        let per_poly_claims: Vec<Vec<BundleClaim<'a, F>>> = rb
            .per_poly_forms
            .iter()
            .zip(&rb.per_poly_values)
            .map(|(forms, values)| {
                forms
                    .iter()
                    .zip(values)
                    .map(|(form, &value)| BundleClaim { form, value })
                    .collect()
            })
            .collect();
        BundleDescriptor {
            num_polys,
            length,
            per_poly_claims,
        }
    }

    /// Helper: run commit → prove → verify → FinalClaim::verify_bundles and
    /// assert the transcript was fully consumed.
    fn run_roundtrip(
        cfg: &BatchedProtocolConfig<Emb>,
        bundles_data: &[RandomBundle],
        bundle_specs: &[BundleSpec],
        label: &str,
    ) {
        let ds = DomainSeparator::protocol(&"zook-batched-test")
            .session(&label.to_string())
            .instance(&Empty);

        // Prover.
        let mut ps = ProverState::new_std(&ds);
        let mut committed = Vec::with_capacity(bundles_data.len());
        let prover_bundles: Vec<WitnessBundle<F>> =
            bundles_data.iter().map(witness_bundle).collect();
        for (i, b) in prover_bundles.iter().enumerate() {
            committed.push(cfg.commit_bundle(&mut ps, i, b));
        }
        let bundle_refs: Vec<&WitnessBundle<F>> = prover_bundles.iter().collect();
        cfg.prove(&mut ps, committed, &bundle_refs);
        let proof = ps.proof();

        // Verifier.
        let mut vs = VerifierState::new_std(&ds, &proof);
        let mut commitments = Vec::with_capacity(bundles_data.len());
        for i in 0..bundles_data.len() {
            commitments.push(cfg.receive_bundle_commitment(&mut vs, i).unwrap());
        }
        let descriptors: Vec<BundleDescriptor<F>> = bundles_data
            .iter()
            .zip(bundle_specs)
            .map(|(rb, spec)| bundle_descriptor(rb, spec.num_polys, spec.length))
            .collect();
        let descriptor_refs: Vec<&BundleDescriptor<F>> = descriptors.iter().collect();
        let claim = cfg.verify(&mut vs, commitments, &descriptor_refs).unwrap();

        claim
            .verify_bundles(&descriptor_refs)
            .expect("FinalClaim::verify_bundles failed");
        vs.check_eof().expect("transcript fully consumed");
    }

    // ---------------------------------------------------------------------------
    // Path A: single bundle, single poly, single claim.
    // ---------------------------------------------------------------------------

    #[test]
    fn roundtrip_single_bundle_single_poly_single_claim_zk() {
        let specs = vec![BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        }];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(1, 8, 1, 0)];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "single_bundle_single_poly_single_claim_zk",
        );
    }

    #[test]
    fn roundtrip_single_bundle_single_poly_single_claim_standard() {
        let specs = vec![BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        }];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::Standard),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(1, 8, 1, 1)];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "single_bundle_single_poly_single_claim_standard",
        );
    }

    #[test]
    fn roundtrip_single_bundle_single_poly_multi_claim_zk() {
        let specs = vec![BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        }];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(1, 8, 3, 2)];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "single_bundle_single_poly_multi_claim_zk",
        );
    }

    // ---------------------------------------------------------------------------
    // Path A: multiple bundles joining at round 0 (same shape).
    // ---------------------------------------------------------------------------

    #[test]
    fn roundtrip_two_bundles_round_zero_zk() {
        let specs = vec![
            BundleSpec {
                length: 1 << 8,
                num_polys: 1,
            };
            2
        ];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(1, 8, 1, 10), random_bundle(1, 8, 1, 11)];
        run_roundtrip(&cfg, &bundles_data, &specs, "two_bundles_round_zero_zk");
    }

    #[test]
    fn roundtrip_three_bundles_round_zero_standard() {
        let specs = vec![
            BundleSpec {
                length: 1 << 8,
                num_polys: 1,
            };
            3
        ];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::Standard),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![
            random_bundle(1, 8, 2, 20),
            random_bundle(1, 8, 1, 21),
            random_bundle(1, 8, 3, 22),
        ];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "three_bundles_round_zero_standard",
        );
    }

    // ---------------------------------------------------------------------------
    // Multi-poly bundles — Axis A (intra-bundle γ-RLC).
    // ---------------------------------------------------------------------------

    #[test]
    fn roundtrip_single_bundle_multi_poly_zk() {
        // 4 polys × 256 each → flat size 1024 matches round-0 source (with
        // folding_factor 2 → inner_vector_size = 4 · 256 = 1024).
        let specs = vec![BundleSpec {
            length: 1 << 8,
            num_polys: 4,
        }];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(4, 8, 1, 40)];
        run_roundtrip(&cfg, &bundles_data, &specs, "single_bundle_multi_poly_zk");
    }

    #[test]
    fn roundtrip_single_bundle_multi_poly_distinct_forms_per_poly_zk() {
        // Each poly carries 2 distinct multilinear-extension forms — exercises
        // the extended-dim trick's per-poly-distinct-claim path.
        let specs = vec![BundleSpec {
            length: 1 << 8,
            num_polys: 2,
        }];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(2, 8, 2, 41)];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "single_bundle_multi_poly_distinct_forms",
        );
    }

    #[test]
    fn roundtrip_two_multi_poly_bundles_round_zero_zk() {
        let specs = vec![
            BundleSpec {
                length: 1 << 6,
                num_polys: 4,
            };
            2
        ];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(4, 6, 1, 50), random_bundle(4, 6, 2, 51)];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "two_multi_poly_bundles_round_zero_zk",
        );
    }

    // ---------------------------------------------------------------------------
    // Multi-poly + different-size converge (combines intra-bundle γ-RLC + merge).
    // ---------------------------------------------------------------------------

    #[test]
    fn roundtrip_multi_poly_diff_sizes_converge_zk() {
        // num_polys = 4 throughout. Δ = 2 = folding factor: bundle 0 (flat 2^10)
        // folds once by 2 to the merge point 2^8; bundle 1 (flat 2^8) is already
        // there → round-0 merge.
        let specs = vec![
            BundleSpec {
                length: 1 << 8,
                num_polys: 4,
            },
            BundleSpec {
                length: 1 << 6,
                num_polys: 4,
            },
        ];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(4, 8, 1, 60), random_bundle(4, 6, 2, 61)];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "multi_poly_diff_sizes_converge_zk",
        );
    }

    // ---------------------------------------------------------------------------
    // Different size, only the bigger folds (smaller already at the merge point).
    // ---------------------------------------------------------------------------

    #[test]
    fn roundtrip_diff_sizes_only_bigger_folds_zk() {
        // Δ = 2 = folding factor: the bigger (2^8) folds once to 2^6, the smaller
        // (2^6) is the merge point → round-0 merge.
        let specs = vec![
            BundleSpec {
                length: 1 << 8,
                num_polys: 1,
            },
            BundleSpec {
                length: 1 << 6,
                num_polys: 1,
            },
        ];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .unwrap();

        let bundles_data = vec![random_bundle(1, 8, 1, 30), random_bundle(1, 6, 1, 31)];
        run_roundtrip(
            &cfg,
            &bundles_data,
            &specs,
            "diff_sizes_only_bigger_folds_zk",
        );
    }

    // ---------------------------------------------------------------------------
    // Different-size bundles: auto-computed one-round converge to a round-0 merge.
    // ---------------------------------------------------------------------------

    /// `(2^10, 2^9)` (Δ = 1 < folding factor 2): each bundle runs one pre-merge
    /// round (2^10 folds by 2, 2^9 by 1) down to a common 2^8, then merges at
    /// round 0. The caller picks nothing — derive computes the folds.
    #[test]
    fn roundtrip_diff_sizes_converge_zk() {
        let specs = vec![
            BundleSpec {
                length: 1 << 10,
                num_polys: 1,
            },
            BundleSpec {
                length: 1 << 9,
                num_polys: 1,
            },
        ];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs.clone()),
        )
        .expect("derive must succeed");

        // Both bundles run exactly one pre-merge round, then merge at round 0.
        assert_eq!(cfg.bundle_configs()[0].pre_merge_rounds.len(), 1);
        assert_eq!(cfg.bundle_configs()[1].pre_merge_rounds.len(), 1);
        assert_eq!(cfg.bundle_configs()[0].join_round, Some(0));
        assert_eq!(cfg.bundle_configs()[1].join_round, Some(0));

        let bundles_data = vec![random_bundle(1, 10, 1, 80), random_bundle(1, 9, 1, 81)];
        run_roundtrip(&cfg, &bundles_data, &specs, "diff_sizes_converge_zk");
    }

    /// Standard-mode variant of the different-size converge roundtrip.
    #[test]
    fn roundtrip_diff_sizes_converge_standard() {
        let specs = vec![
            BundleSpec {
                length: 1 << 10,
                num_polys: 1,
            },
            BundleSpec {
                length: 1 << 9,
                num_polys: 1,
            },
        ];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::Standard),
            batched_tuning(specs.clone()),
        )
        .expect("derive must succeed");

        let bundles_data = vec![random_bundle(1, 10, 1, 93), random_bundle(1, 9, 1, 94)];
        run_roundtrip(&cfg, &bundles_data, &specs, "diff_sizes_converge_standard");
    }

    // ---------------------------------------------------------------------------
    // Negative tests: tamper detection.
    // ---------------------------------------------------------------------------

    #[test]
    fn verify_rejects_wrong_value_zk() {
        let specs = vec![BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        }];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs),
        )
        .unwrap();
        let mut bundles_data = [random_bundle(1, 8, 1, 100)];

        // Build proof against TRUE claims.
        let ds = DomainSeparator::protocol(&"zook-batched-test")
            .session(&"wrong_value".to_string())
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let prover_bundles: Vec<WitnessBundle<F>> =
            bundles_data.iter().map(witness_bundle).collect();
        let committed = vec![cfg.commit_bundle(&mut ps, 0, &prover_bundles[0])];
        let bundle_refs: Vec<&WitnessBundle<F>> = prover_bundles.iter().collect();
        cfg.prove(&mut ps, committed, &bundle_refs);
        let proof = ps.proof();

        // CORRUPT the verifier-side value AFTER proving — verify must reject.
        bundles_data[0].per_poly_values[0][0] += F::ONE;

        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitments = vec![cfg.receive_bundle_commitment(&mut vs, 0).unwrap()];
        let descriptors = [bundle_descriptor(&bundles_data[0], 1, 1 << 8)];
        let descriptor_refs: Vec<&BundleDescriptor<F>> = descriptors.iter().collect();

        let verify_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let claim = cfg.verify(&mut vs, commitments, &descriptor_refs)?;
            claim.verify_bundles(&descriptor_refs)
        }));
        match verify_result {
            Err(_) | Ok(Err(_)) => {} // expected: failed
            Ok(Ok(())) => panic!("expected verification to fail on corrupted value"),
        }
    }

    #[test]
    fn verify_rejects_malformed_descriptor_without_panic() {
        let specs = vec![BundleSpec {
            length: 1 << 8,
            num_polys: 1,
        }];
        let cfg = BatchedProtocolConfig::<Emb>::derive(
            test_spec(Mode::ZeroKnowledge),
            batched_tuning(specs),
        )
        .unwrap();
        let bundles_data = [random_bundle(1, 8, 1, 101)];

        let ds = DomainSeparator::protocol(&"zook-batched-test")
            .session(&"malformed_descriptor".to_string())
            .instance(&Empty);
        let mut ps = ProverState::new_std(&ds);
        let prover_bundles: Vec<WitnessBundle<F>> =
            bundles_data.iter().map(witness_bundle).collect();
        let committed = vec![cfg.commit_bundle(&mut ps, 0, &prover_bundles[0])];
        let bundle_refs: Vec<&WitnessBundle<F>> = prover_bundles.iter().collect();
        cfg.prove(&mut ps, committed, &bundle_refs);
        let proof = ps.proof();

        let mut vs = VerifierState::new_std(&ds, &proof);
        let commitments = vec![cfg.receive_bundle_commitment(&mut vs, 0).unwrap()];
        let mut descriptors = [bundle_descriptor(&bundles_data[0], 1, 1 << 8)];
        descriptors[0].per_poly_claims.push(Vec::new());
        let descriptor_refs: Vec<&BundleDescriptor<F>> = descriptors.iter().collect();

        let verify_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cfg.verify(&mut vs, commitments, &descriptor_refs)
        }));
        match verify_result {
            // Rejected either by returning `Err` (normal build) or by the
            // `verify!` check panicking (verifier_panics feature). Both are
            // correct rejections; only silent acceptance is a bug.
            Err(_) | Ok(Err(_)) => {}
            Ok(Ok(_)) => panic!("expected malformed descriptor to be rejected"),
        }
    }
}
