//! Intra-bundle γ-RLC: `N` same-size polys committed together, batched into one virtual block
//! via the extended-dimension coordinate trick (treats the bundle as one `(d + log N)`-variate poly).

use ark_ff::Field;

use crate::{algebra::linear_form::LinearForm, bits::Bits, transcript::VerificationResult, verify};

/// One claim attached to a polynomial in the bundle: `<poly, form> = value`.
pub struct BundleClaim<'a, F: Field> {
    pub form: &'a dyn LinearForm<F>,
    pub value: F,
}

/// User-facing bundle: `N` polynomials of identical length plus per-poly claim lists.
pub struct WitnessBundle<'a, F: Field> {
    pub polys: Vec<&'a [F]>,
    pub per_poly_claims: Vec<Vec<BundleClaim<'a, F>>>,
}

/// Verifier-side view of a bundle: same shape as [`WitnessBundle`] without the witness data.
pub struct BundleDescriptor<'a, F: Field> {
    pub num_polys: usize,
    pub length: usize,
    pub per_poly_claims: Vec<Vec<BundleClaim<'a, F>>>,
}

impl<F: Field> BundleDescriptor<'_, F> {
    /// Validate verifier-facing structural invariants.
    pub fn validate(&self) -> VerificationResult<()> {
        verify!(self.num_polys.is_power_of_two());
        verify!(self.length.is_power_of_two());
        verify!(self.per_poly_claims.len() == self.num_polys);
        for claims in &self.per_poly_claims {
            for claim in claims {
                verify!(claim.form.size() == self.length);
            }
        }
        Ok(())
    }

    /// Total number of `(poly, claim)` pairs across the bundle.
    pub fn total_claims(&self) -> usize {
        self.per_poly_claims.iter().map(Vec::len).sum()
    }

    /// log₂ of `num_polys × length`.
    pub const fn domain_bits(&self) -> usize {
        self.num_polys.trailing_zeros() as usize + self.length.trailing_zeros() as usize
    }

    /// log₂ of `num_polys` (high bits of the eval point — poly axis).
    pub const fn log_num_polys(&self) -> usize {
        self.num_polys.trailing_zeros() as usize
    }

    /// log₂ of `length` (low bits of the eval point — message axis).
    pub const fn log_length(&self) -> usize {
        self.length.trailing_zeros() as usize
    }

    /// Bundle's initial sum `s = Σ_{k, j} γ^{idx(k, j)} · v_{k, j}` from claim values.
    pub fn initial_sum(&self, batching_challenge: F) -> F {
        let total = self.total_claims();
        if total == 0 {
            return F::ZERO;
        }
        let mut claim_weights = Vec::with_capacity(total);
        claim_weights.push(F::ONE);
        for _ in 1..total {
            let next = *claim_weights.last().unwrap() * batching_challenge;
            claim_weights.push(next);
        }
        let mut idx = 0;
        let mut s = F::ZERO;
        for claims in &self.per_poly_claims {
            for claim in claims {
                s += claim_weights[idx] * claim.value;
                idx += 1;
            }
        }
        s
    }
}

impl<F: Field> WitnessBundle<'_, F> {
    /// Number of polys `N`.
    pub const fn num_polys(&self) -> usize {
        self.polys.len()
    }

    /// Per-poly length `M`. Returns 0 if the bundle is empty.
    pub fn poly_len(&self) -> usize {
        self.polys.first().map_or(0, |p| p.len())
    }

    /// Total number of `(poly, claim)` pairs in the bundle.
    pub fn total_claims(&self) -> usize {
        self.per_poly_claims.iter().map(Vec::len).sum()
    }

    /// Assert same-size invariant and structural consistency. Power-of-two checks
    /// are owned by the downstream IRS commit, not here.
    pub fn assert_well_formed(&self) {
        assert_eq!(
            self.polys.len(),
            self.per_poly_claims.len(),
            "polys.len() must equal per_poly_claims.len()"
        );
        if self.polys.is_empty() {
            return;
        }
        let m = self.poly_len();
        for (k, p) in self.polys.iter().enumerate() {
            assert_eq!(
                p.len(),
                m,
                "all polys in a bundle must share the same length; polys[{k}].len() != polys[0].len()"
            );
        }
        for (k, claims) in self.per_poly_claims.iter().enumerate() {
            for (j, claim) in claims.iter().enumerate() {
                assert_eq!(
                    claim.form.size(),
                    m,
                    "per_poly_claims[{k}][{j}].form.size() must match poly length"
                );
            }
        }
    }
}

/// Concatenate the bundle's polys into one length-`N · M` vector; high bits index the poly.
pub fn flatten_polys<F: Field>(bundle: &WitnessBundle<F>) -> Vec<F> {
    let n = bundle.num_polys();
    let m = bundle.poly_len();
    let mut out = Vec::with_capacity(n * m);
    for p in &bundle.polys {
        debug_assert_eq!(p.len(), m);
        out.extend_from_slice(p);
    }
    out
}

/// γ-RLC reduction of a bundle into one virtual block (combined covector + sum + γ powers).
pub fn build_bundle_claim<F: Field>(
    bundle: &WitnessBundle<F>,
    batching_challenge: F,
) -> BundleClaimMaterial<F> {
    bundle.assert_well_formed();
    let n = bundle.num_polys();
    let m = bundle.poly_len();
    let total = bundle.total_claims();

    let mut claim_weights = Vec::with_capacity(total);
    if total > 0 {
        claim_weights.push(F::ONE);
        for _ in 1..total {
            let next = *claim_weights.last().unwrap() * batching_challenge;
            claim_weights.push(next);
        }
    }

    let mut covector = vec![F::ZERO; n * m];
    let mut sum = F::ZERO;
    let mut idx = 0;
    for (k, claims) in bundle.per_poly_claims.iter().enumerate() {
        let poly_slice = &mut covector[k * m..(k + 1) * m];
        for claim in claims {
            let weight = claim_weights[idx];
            claim.form.accumulate(poly_slice, weight);
            sum += weight * claim.value;
            idx += 1;
        }
    }
    debug_assert_eq!(idx, total);

    BundleClaimMaterial {
        covector,
        sum,
        claim_weights,
    }
}

/// Output of [`build_bundle_claim`].
pub struct BundleClaimMaterial<F: Field> {
    pub covector: Vec<F>,
    pub sum: F,
    pub claim_weights: Vec<F>,
}

/// Per-bundle γ-RLC soundness in bits via Schwartz–Zippel on a `(total_claims − 1)`-degree
/// error polynomial. Returns `field_bits` when `total_claims ≤ 1` so callers can compose finite bits.
pub fn analytic_error_bits(total_claims: usize, field_bits: f64) -> Bits {
    if total_claims <= 1 {
        return Bits::new(field_bits);
    }
    let log_factor = ((total_claims - 1) as f64).log2();
    Bits::new(field_bits - log_factor)
}

#[cfg(test)]
mod tests {
    use ark_ff::AdditiveGroup;
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};

    use super::*;
    use crate::algebra::{
        dot,
        embedding::Identity,
        fields::Field64,
        linear_form::{Evaluate, MultilinearExtension},
        random_vector,
    };

    type F = Field64;
    type Emb = Identity<F>;
    /// `(polys, per_poly_claims)` produced by [`build_random_bundle`].
    type RandomBundle = (Vec<Vec<F>>, Vec<Vec<(MultilinearExtension<F>, F)>>);

    fn build_random_bundle(n: usize, d: usize, claims_per_poly: usize, seed: u64) -> RandomBundle {
        let embedding = Emb::default();
        let mut rng = StdRng::seed_from_u64(seed);
        let m = 1usize << d;
        let polys: Vec<Vec<F>> = (0..n).map(|_| random_vector(&mut rng, m)).collect();
        let mut per_poly_claims: Vec<Vec<(MultilinearExtension<F>, F)>> = Vec::with_capacity(n);
        for poly in &polys {
            let mut claims = Vec::with_capacity(claims_per_poly);
            for _ in 0..claims_per_poly {
                let point: Vec<F> = random_vector(&mut rng, d);
                let form = MultilinearExtension { point };
                let value = form.evaluate(&embedding, poly);
                claims.push((form, value));
            }
            per_poly_claims.push(claims);
        }
        (polys, per_poly_claims)
    }

    fn bundle_view<'a>(
        polys: &'a [Vec<F>],
        per_poly_claims: &'a [Vec<(MultilinearExtension<F>, F)>],
    ) -> WitnessBundle<'a, F> {
        let polys_slices: Vec<&[F]> = polys.iter().map(Vec::as_slice).collect();
        let claims: Vec<Vec<BundleClaim<'a, F>>> = per_poly_claims
            .iter()
            .map(|cl| {
                cl.iter()
                    .map(|(form, value)| BundleClaim {
                        form: form as &dyn LinearForm<F>,
                        value: *value,
                    })
                    .collect()
            })
            .collect();
        WitnessBundle {
            polys: polys_slices,
            per_poly_claims: claims,
        }
    }

    #[test]
    fn flatten_concatenates_high_bits_poly_index() {
        let polys = vec![
            vec![F::from(1u64), F::from(2)],
            vec![F::from(3), F::from(4)],
        ];
        let claims: Vec<Vec<(MultilinearExtension<F>, F)>> = vec![vec![], vec![]];
        let bundle = bundle_view(&polys, &claims);
        let flat = flatten_polys(&bundle);
        assert_eq!(
            flat,
            vec![F::from(1u64), F::from(2), F::from(3), F::from(4)]
        );
    }

    #[test]
    fn inner_product_identity_holds_distinct_forms_per_poly() {
        let mut rng = StdRng::seed_from_u64(42);
        for (n, d, claims) in [(1, 3, 2), (2, 4, 1), (3, 5, 3), (4, 2, 2)] {
            let (polys, per_poly_claims) = build_random_bundle(n, d, claims, rng.gen());
            let bundle = bundle_view(&polys, &per_poly_claims);
            let flat = flatten_polys(&bundle);
            let batching_challenge: F = rng.gen();
            let reduced_claim = build_bundle_claim(&bundle, batching_challenge);
            assert_eq!(
                reduced_claim.covector.len(),
                bundle.num_polys() * bundle.poly_len()
            );
            assert_eq!(dot(&flat, &reduced_claim.covector), reduced_claim.sum);
        }
    }

    #[test]
    fn altered_value_breaks_the_identity() {
        let mut rng = StdRng::seed_from_u64(7);
        let (polys, mut per_poly_claims) = build_random_bundle(3, 4, 2, rng.gen());
        per_poly_claims[1][0].1 += F::ONE;
        let bundle = bundle_view(&polys, &per_poly_claims);
        let flat = flatten_polys(&bundle);
        let batching_challenge: F = rng.gen();
        let reduced_claim = build_bundle_claim(&bundle, batching_challenge);
        assert_ne!(dot(&flat, &reduced_claim.covector), reduced_claim.sum);
    }

    #[test]
    fn no_claims_bundle_has_zero_sum_zero_covector() {
        let mut rng = StdRng::seed_from_u64(0);
        let polys = vec![random_vector::<F>(&mut rng, 4); 2];
        let per_poly_claims: Vec<Vec<(MultilinearExtension<F>, F)>> = vec![vec![], vec![]];
        let bundle = bundle_view(&polys, &per_poly_claims);
        let reduced_claim = build_bundle_claim(&bundle, F::from(7u64));
        assert_eq!(reduced_claim.sum, F::ZERO);
        assert!(reduced_claim.covector.iter().all(|&x| x == F::ZERO));
        assert!(reduced_claim.claim_weights.is_empty());
    }

    #[test]
    fn claim_weights_are_geometric_sequence() {
        let mut rng = StdRng::seed_from_u64(99);
        let (polys, per_poly_claims) = build_random_bundle(2, 3, 3, rng.gen());
        let bundle = bundle_view(&polys, &per_poly_claims);
        let batching_challenge: F = rng.gen();
        let reduced_claim = build_bundle_claim(&bundle, batching_challenge);
        assert_eq!(reduced_claim.claim_weights.len(), bundle.total_claims());
        assert_eq!(reduced_claim.claim_weights[0], F::ONE);
        for w in reduced_claim.claim_weights.windows(2) {
            assert_eq!(w[1], w[0] * batching_challenge);
        }
    }
}
