//! Field algebra: embeddings, field definitions, linear forms, multilinear
//! polynomials, NTTs, sumcheck kernels, and dense vector arithmetic.

pub mod embedding;
pub mod fields;
pub mod linear_form;
mod multilinear;
pub mod ntt;
pub mod sumcheck_kernels;
mod vector;

pub use multilinear::{
    eq_weights, eval_eq, fold_chunks, mixed_fold_chunks, mixed_multilinear_extend,
    multilinear_extend,
};
pub(crate) use vector::geometric_accumulate_prefixes;
pub use vector::{
    dot, geometric_accumulate, geometric_sequence, lift, mixed_dot, mixed_scalar_mul_add,
    mixed_univariate_evaluate, random_vector, scalar_mul, scalar_mul_add, scalar_mul_add_new,
    univariate_evaluate,
};
