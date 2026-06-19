// NOTE: 100% AI GENERATED

use std::any::type_name;

use ark_ff::{FftField, Field};

use crate::{
    algebra::{
        fields::Field256,
        ntt::{Messages, NttEngine, ReedSolomon},
    },
    buffer::{BufferOps, MetalBuffer},
};

use super::runtime::{assert_bn254, encode_single_vector_coset_ntt};

const RS_PRIMES: [usize; 2] = [2, 3];

fn rs_divisors(n: usize) -> Vec<usize> {
    let mut result = vec![1usize];
    let mut remaining = n;
    for &p in &RS_PRIMES {
        let mut pk = 1usize;
        let existing = result.clone();
        while remaining.is_multiple_of(p) {
            pk *= p;
            remaining /= p;
            result.extend(existing.iter().map(|d| d * pk));
        }
    }
    assert_eq!(remaining, 1);
    result.sort_unstable();
    result
}

#[cfg(not(feature = "rs_in_order"))]
pub(crate) fn rs_transpose_permute(index: usize, rows: usize, cols: usize) -> usize {
    debug_assert!(index < rows * cols);
    let (row, col) = (index / cols, index % cols);
    row + col * rows
}
/// Metal (GPU) Reed-Solomon encoder.
///
/// Keeps the scalar Reed-Solomon parameters directly and runs encoding on Metal buffers.
#[derive(Debug, Clone)]
pub struct MetalRs<F: Field> {
    order: usize,
    divisors: Vec<usize>,
    omega_order: F,
}

impl<F: Field> MetalRs<F> {
    pub fn new(order: usize, omega_order: F) -> Self {
        assert_eq!(omega_order.pow([order as u64]), F::ONE);
        for prime in RS_PRIMES {
            if order.is_multiple_of(prime) {
                assert_ne!(omega_order.pow([(order / prime) as u64]), F::ONE);
            }
        }
        Self {
            order,
            divisors: rs_divisors(order),
            omega_order,
        }
    }
}

impl<F: FftField> MetalRs<F> {
    pub fn new_from_fftfield() -> Self {
        let (mut omega, mut order) = if let (Some(mut omega), Some(b), Some(k)) = (
            F::LARGE_SUBGROUP_ROOT_OF_UNITY,
            F::SMALL_SUBGROUP_BASE,
            F::SMALL_SUBGROUP_BASE_ADICITY,
        ) {
            let mut order = 1;
            let mut remaining = (b as usize).checked_pow(k).expect("Small group too large.");
            for p in RS_PRIMES {
                while remaining.is_multiple_of(p) {
                    order *= p;
                    remaining /= p;
                }
            }
            omega = omega.pow([remaining as u64]);
            (omega, order)
        } else {
            (F::TWO_ADIC_ROOT_OF_UNITY, 1)
        };
        let twos = F::TWO_ADICITY.min(order.leading_zeros()) as usize;
        for _ in 0..(F::TWO_ADICITY as usize - twos) {
            omega.square_in_place();
        }
        order <<= twos;
        Self::new(order, omega)
    }
}

impl<F: Field> ReedSolomon<F> for MetalRs<F> {
    fn next_order(&self, size: usize) -> Option<usize> {
        match self.divisors.binary_search(&size) {
            Ok(index) | Err(index) => self.divisors.get(index).copied(),
        }
    }

    fn generator(&self, codeword_length: usize) -> F {
        self.omega_order
            .pow([(self.order / codeword_length) as u64])
    }

    fn evaluation_points(
        &self,
        masked_message_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<F> {
        assert!(masked_message_length <= codeword_length);
        assert!(self.order.is_multiple_of(codeword_length));
        let mut result = Vec::new();
        let generator = self.generator(codeword_length);

        let mut coset_size = self.next_order(masked_message_length).unwrap();
        while !codeword_length.is_multiple_of(coset_size) {
            coset_size = self.next_order(coset_size + 1).unwrap();
        }
        let num_cosets = codeword_length / coset_size;
        #[cfg(feature = "rs_in_order")]
        let _ = (coset_size, num_cosets);

        for &index in indices {
            assert!(index < codeword_length);

            #[cfg(not(feature = "rs_in_order"))]
            let index = rs_transpose_permute(index, num_cosets, coset_size);
            result.push(generator.pow([index as u64]));
        }
        result
    }

    fn interleaved_encode(
        &self,
        messages: Messages<'_, F>,
        masks: &MetalBuffer<F>,
        codeword_length: usize,
    ) -> MetalBuffer<F> {
        let vectors = messages.vectors;
        let num_messages = vectors.len() * messages.interleaving_depth;
        if num_messages == 0 {
            return BufferOps::from_vec(Vec::new());
        }
        assert!(masks.len().is_multiple_of(num_messages));
        let mask_length = masks.len() / num_messages;

        if type_name::<F>() == type_name::<Field256>() && vectors.len() == 1 && mask_length == 0 {
            assert_bn254::<F>();
            let mut coset_size = self.next_order(messages.message_length).unwrap();
            while !codeword_length.is_multiple_of(coset_size) {
                coset_size = self.next_order(coset_size + 1).unwrap();
            }
            return encode_single_vector_coset_ntt(
                vectors[0],
                messages.message_length,
                messages.interleaving_depth,
                codeword_length,
                coset_size,
            );
        }

        NttEngine::new(self.order, self.omega_order).interleaved_encode(
            messages,
            masks,
            codeword_length,
        )
    }
}
