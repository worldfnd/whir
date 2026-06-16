use std::{
    any::type_name,
    cell::OnceCell,
    cmp::Ordering,
    collections::HashMap,
    hash::{Hash, Hasher},
    marker::PhantomData,
    os::raw::c_void,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use ark_ff::{AdditiveGroup, BigInt, FftField, Field, Fp, MontBackend};
use ark_std::rand::{distributions::Standard, prelude::Distribution, CryptoRng, Rng, RngCore};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use zerocopy::IntoBytes;

use crate::{
    algebra::{
        buffer::{BufferOps, FieldOps},
        embedding::{Embedding, Identity},
        fields::{BN254Config, Field256},
        linear_form::{Covector, LinearForm, UnivariateEvaluation},
        ntt::{ReedSolomon, RsDomain},
    },
    hash::{metal_profile, Hash as Digest, MetalSha2},
};

const METAL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct F { ulong v[4]; };

constant ulong MODULUS[4] = {
    0x43e1f593f0000001UL,
    0x2833e84879b97091UL,
    0xb85045b68181585dUL,
    0x30644e72e131a029UL,
};

constant ulong BN254_N0PRIME = 0xc2e1f593efffffffUL;
constant F CANONICAL_ONE = {{1UL, 0UL, 0UL, 0UL}};
constant F MONT_ONE = {{
    0xac96341c4ffffffbUL,
    0x36fc76959f60cd29UL,
    0x666ea36f7879462eUL,
    0xe0a77c19a07df2fUL,
}};

struct StageConfig {
    uint row_len;
    uint half_m;
    uint twiddle_offset;
    uint _pad0;
};

struct BitReverseParams {
    uint row_len;
    uint log_n;
    uint total_elements;
    uint _pad0;
};

struct TransposeParams {
    uint rows;
    uint cols;
    uint total_elements;
};

struct ReplicateCosetsParams {
    uint row_len;
    uint coset_size;
    uint trailing_elements;
};

struct PackSingleVectorParams {
    uint row_count;
    uint message_length;
    uint codeword_length;
    uint coset_size;
    uint total_elements;
};

struct FieldBytesParams {
    uint rows;
    uint cols;
};

struct ApplyCosetTwiddlesParams {
    uint row_count;
    uint num_cosets;
    uint coset_size;
    uint codeword_length;
    uint total_elements;
};

inline F zero_f() {
    F r;
    r.v[0] = 0; r.v[1] = 0; r.v[2] = 0; r.v[3] = 0;
    return r;
}

inline F one_f() {
    return MONT_ONE;
}

inline F load_f(device const ulong *data, uint idx) {
    F r;
    uint base = idx * 4;
    r.v[0] = data[base + 0];
    r.v[1] = data[base + 1];
    r.v[2] = data[base + 2];
    r.v[3] = data[base + 3];
    return r;
}

inline void store_f(device ulong *data, uint idx, F x) {
    uint base = idx * 4;
    data[base + 0] = x.v[0];
    data[base + 1] = x.v[1];
    data[base + 2] = x.v[2];
    data[base + 3] = x.v[3];
}

inline bool ge_mod(F a) {
    for (int i = 3; i >= 0; i--) {
        if (a.v[i] > MODULUS[i]) return true;
        if (a.v[i] < MODULUS[i]) return false;
    }
    return true;
}

inline bool ge_f(F a, F b) {
    for (int i = 3; i >= 0; i--) {
        if (a.v[i] > b.v[i]) return true;
        if (a.v[i] < b.v[i]) return false;
    }
    return true;
}

inline F sub_raw(F a, thread const ulong *b, thread bool &borrow) {
    F r;
    borrow = false;
    for (uint i = 0; i < 4; i++) {
        ulong bi = b[i] + (borrow ? 1UL : 0UL);
        bool bcarry = borrow && bi == 0;
        r.v[i] = a.v[i] - bi;
        borrow = bcarry || a.v[i] < bi;
    }
    return r;
}

inline F sub_modulus(F a) {
    F r;
    bool borrow = false;
    for (uint i = 0; i < 4; i++) {
        ulong bi = MODULUS[i] + (borrow ? 1UL : 0UL);
        bool bcarry = borrow && bi == 0;
        r.v[i] = a.v[i] - bi;
        borrow = bcarry || a.v[i] < bi;
    }
    return r;
}

inline F add_modulus(F a) {
    F r;
    bool carry = false;
    for (uint i = 0; i < 4; i++) {
        ulong mi = MODULUS[i];
        ulong sum = a.v[i] + mi;
        bool c0 = sum < a.v[i];
        ulong sum2 = sum + (carry ? 1UL : 0UL);
        bool c1 = carry && sum2 == 0;
        r.v[i] = sum2;
        carry = c0 || c1;
    }
    return r;
}

inline F add_f(F a, F b) {
    F r;
    bool carry = false;
    for (uint i = 0; i < 4; i++) {
        ulong sum = a.v[i] + b.v[i];
        bool c0 = sum < a.v[i];
        ulong sum2 = sum + (carry ? 1UL : 0UL);
        bool c1 = carry && sum2 == 0;
        r.v[i] = sum2;
        carry = c0 || c1;
    }
    if (carry || ge_mod(r)) {
        r = sub_modulus(r);
    }
    return r;
}

inline F sub_f(F a, F b) {
    ulong bv[4] = { b.v[0], b.v[1], b.v[2], b.v[3] };
    bool borrow = false;
    F r = sub_raw(a, bv, borrow);
    if (borrow) {
        r = add_modulus(r);
    }
    return r;
}

inline F double_f(F a) {
    return add_f(a, a);
}

inline ulong add_with_carry(ulong a, ulong b, thread ulong &carry) {
    ulong sum = a + b;
    ulong c1 = sum < a ? 1UL : 0UL;
    ulong sum_with_carry = sum + carry;
    ulong c2 = sum_with_carry < sum ? 1UL : 0UL;
    carry = c1 + c2;
    return sum_with_carry;
}

inline void add_scaled_step(thread ulong &dst, ulong s, ulong a, thread ulong &carry) {
    ulong product_lo = s * a;
    ulong product_hi = mulhi(s, a);

    ulong sum = dst + product_lo;
    ulong carry0 = sum < dst ? 1UL : 0UL;
    ulong sum_with_carry = sum + carry;
    ulong carry1 = sum_with_carry < sum ? 1UL : 0UL;

    dst = sum_with_carry;
    carry = product_hi + carry0 + carry1;
}

inline void add_scaled(thread ulong *dst, ulong s, ulong a0, ulong a1, ulong a2, ulong a3) {
    ulong carry = 0;
    add_scaled_step(dst[0], s, a0, carry);
    add_scaled_step(dst[1], s, a1, carry);
    add_scaled_step(dst[2], s, a2, carry);
    add_scaled_step(dst[3], s, a3, carry);
    dst[4] += carry;
}

inline F mont_mul(F lhs, F rhs) {
    ulong buf[9] = {0};
    uint off = 0;

#pragma clang loop unroll(enable)
    for (uint i = 0; i < 4; i++) {
        add_scaled(&buf[off], lhs.v[i], rhs.v[0], rhs.v[1], rhs.v[2], rhs.v[3]);

        ulong m = buf[off] * BN254_N0PRIME;
        add_scaled(
            &buf[off],
            m,
            MODULUS[0],
            MODULUS[1],
            MODULUS[2],
            MODULUS[3]
        );

        off += 1;
        buf[off + 4] = 0;
    }

    F result;
    result.v[0] = buf[off + 0];
    result.v[1] = buf[off + 1];
    result.v[2] = buf[off + 2];
    result.v[3] = buf[off + 3];
    if (ge_mod(result)) {
        result = sub_modulus(result);
    }
    return result;
}

inline F from_mont(F value) {
    F result = mont_mul(value, CANONICAL_ONE);
    if (ge_mod(result)) {
        result = sub_modulus(result);
    }
    return result;
}

inline F mul_f(F a, F b) {
    return mont_mul(a, b);
}

inline F pow_f(F base, uint exp) {
    F acc = one_f();
    F x = base;
    while (exp != 0) {
        if ((exp & 1) != 0) {
            acc = mul_f(acc, x);
        }
        exp >>= 1;
        if (exp != 0) {
            x = mul_f(x, x);
        }
    }
    return acc;
}

kernel void bn254_fold(
    device ulong *values [[buffer(0)]],
    device const ulong *weight_buf [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    constant uint &fold_half [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= fold_half) return;
    F low = gid < len ? load_f(values, gid) : zero_f();
    F high = gid + fold_half < len ? load_f(values, gid + fold_half) : zero_f();
    F weight = load_f(weight_buf, 0);
    store_f(values, gid, add_f(low, mul_f(sub_f(high, low), weight)));
}

inline F fold_value_at(device const ulong *values, uint len, uint fold_half, uint idx, F weight) {
    F low = idx < len ? load_f(values, idx) : zero_f();
    F high = idx + fold_half < len ? load_f(values, idx + fold_half) : zero_f();
    return add_f(low, mul_f(sub_f(high, low), weight));
}

kernel void bn254_fold_pair(
    device ulong *a [[buffer(0)]],
    device ulong *b [[buffer(1)]],
    device const ulong *weight_buf [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &fold_half [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= fold_half) return;
    F weight = load_f(weight_buf, 0);
    store_f(a, gid, fold_value_at(a, len, fold_half, gid, weight));
    store_f(b, gid, fold_value_at(b, len, fold_half, gid, weight));
}

kernel void bn254_scalar_mul_add(
    device ulong *acc [[buffer(0)]],
    device const ulong *vector [[buffer(1)]],
    device const ulong *weight_buf [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    F weight = load_f(weight_buf, 0);
    store_f(acc, gid, add_f(load_f(acc, gid), mul_f(weight, load_f(vector, gid))));
}

kernel void bn254_dot(
    device const ulong *a [[buffer(0)]],
    device const ulong *b [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    F acc = zero_f();
    for (uint i = 0; i < len; i++) {
        acc = add_f(acc, mul_f(load_f(a, i), load_f(b, i)));
    }
    store_f(out, 0, acc);
}

kernel void bn254_sumcheck(
    device const ulong *a [[buffer(0)]],
    device const ulong *b [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &fold_half [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    F c0 = zero_f();
    F c2 = zero_f();
    for (uint i = 0; i < fold_half; i++) {
        F a0 = i < len ? load_f(a, i) : zero_f();
        F b0 = i < len ? load_f(b, i) : zero_f();
        F a1 = i + fold_half < len ? load_f(a, i + fold_half) : zero_f();
        F b1 = i + fold_half < len ? load_f(b, i + fold_half) : zero_f();
        c0 = add_f(c0, mul_f(a0, b0));
        c2 = add_f(c2, mul_f(sub_f(a1, a0), sub_f(b1, b0)));
    }
    store_f(out, 0, c0);
    store_f(out, 1, c2);
}

kernel void bn254_dot_chunks(
    device const ulong *a [[buffer(0)]],
    device const ulong *b [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &chunk_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * chunk_size;
    if (start >= len) return;
    uint end = min(start + chunk_size, len);
    F acc = zero_f();
    for (uint i = start; i < end; i++) {
        acc = add_f(acc, mul_f(load_f(a, i), load_f(b, i)));
    }
    store_f(out, gid, acc);
}

kernel void bn254_sum_chunks(
    device const ulong *input [[buffer(0)]],
    device ulong *out [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    constant uint &offset [[buffer(3)]],
    constant uint &chunk_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * chunk_size;
    if (start >= len) return;
    uint end = min(start + chunk_size, len);
    F acc = zero_f();
    for (uint i = start; i < end; i++) {
        acc = add_f(acc, load_f(input, offset + i));
    }
    store_f(out, gid, acc);
}

kernel void bn254_sumcheck_reduce_chunks(
    device const ulong *input [[buffer(0)]],
    device ulong *out [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    constant uint &chunk_size [[buffer(3)]],
    constant uint &out_len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= out_len) return;
    uint start = gid * chunk_size;
    uint end = min(start + chunk_size, len);
    F c0 = zero_f();
    F c2 = zero_f();
    for (uint i = start; i < end; i++) {
        c0 = add_f(c0, load_f(input, i));
        c2 = add_f(c2, load_f(input, len + i));
    }
    store_f(out, gid, c0);
    store_f(out, out_len + gid, c2);
}

kernel void bn254_sumcheck_chunks(
    device const ulong *a [[buffer(0)]],
    device const ulong *b [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &fold_half [[buffer(4)]],
    constant uint &chunk_size [[buffer(5)]],
    constant uint &partial_count [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * chunk_size;
    if (start >= fold_half) return;
    uint end = min(start + chunk_size, fold_half);
    F c0 = zero_f();
    F c2 = zero_f();
    for (uint i = start; i < end; i++) {
        F a0 = i < len ? load_f(a, i) : zero_f();
        F b0 = i < len ? load_f(b, i) : zero_f();
        F a1 = i + fold_half < len ? load_f(a, i + fold_half) : zero_f();
        F b1 = i + fold_half < len ? load_f(b, i + fold_half) : zero_f();
        c0 = add_f(c0, mul_f(a0, b0));
        c2 = add_f(c2, mul_f(sub_f(a1, a0), sub_f(b1, b0)));
    }
    store_f(out, gid, c0);
    store_f(out, partial_count + gid, c2);
}

kernel void bn254_fold_pair_sumcheck_chunks(
    device ulong *a [[buffer(0)]],
    device ulong *b [[buffer(1)]],
    device const ulong *weight_buf [[buffer(2)]],
    device ulong *out [[buffer(3)]],
    constant uint &len [[buffer(4)]],
    constant uint &fold_half [[buffer(5)]],
    constant uint &sum_half [[buffer(6)]],
    constant uint &chunk_size [[buffer(7)]],
    constant uint &partial_count [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * chunk_size;
    if (start >= sum_half) return;
    uint end = min(start + chunk_size, sum_half);
    F weight = load_f(weight_buf, 0);
    F c0 = zero_f();
    F c2 = zero_f();
    for (uint i = start; i < end; i++) {
        uint right = i + sum_half;
        F a0 = fold_value_at(a, len, fold_half, i, weight);
        F b0 = fold_value_at(b, len, fold_half, i, weight);
        F a1 = right < fold_half ? fold_value_at(a, len, fold_half, right, weight) : zero_f();
        F b1 = right < fold_half ? fold_value_at(b, len, fold_half, right, weight) : zero_f();
        store_f(a, i, a0);
        store_f(b, i, b0);
        if (right < fold_half) {
            store_f(a, right, a1);
            store_f(b, right, b1);
        }
        c0 = add_f(c0, mul_f(a0, b0));
        c2 = add_f(c2, mul_f(sub_f(a1, a0), sub_f(b1, b0)));
    }
    store_f(out, gid, c0);
    store_f(out, partial_count + gid, c2);
}

kernel void bn254_geometric_accumulate(
    device ulong *acc [[buffer(0)]],
    device const ulong *points [[buffer(1)]],
    device const ulong *scalars [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &num_points [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    F value = load_f(acc, gid);
    for (uint j = 0; j < num_points; j++) {
        value = add_f(value, mul_f(load_f(scalars, j), pow_f(load_f(points, j), gid)));
    }
    store_f(acc, gid, value);
}

kernel void bn254_geometric_accumulate_chunks(
    device ulong *acc [[buffer(0)]],
    device const ulong *points [[buffer(1)]],
    device const ulong *scalars [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &num_points [[buffer(4)]],
    constant uint &chunk_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * chunk_size;
    if (start >= len) return;
    uint end = min(start + chunk_size, len);
    for (uint j = 0; j < num_points; j++) {
        F point = load_f(points, j);
        F scalar = load_f(scalars, j);
        F power = pow_f(point, start);
        for (uint i = start; i < end; i++) {
            F value = load_f(acc, i);
            value = add_f(value, mul_f(scalar, power));
            store_f(acc, i, value);
            power = mul_f(power, point);
        }
    }
}

kernel void bn254_geometric_accumulate_chunks_strided(
    device ulong *acc [[buffer(0)]],
    device const ulong *points [[buffer(1)]],
    device const ulong *point_steps [[buffer(2)]],
    device const ulong *scalars [[buffer(3)]],
    constant uint &len [[buffer(4)]],
    constant uint &num_points [[buffer(5)]],
    constant uint &chunk_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * chunk_size;
    if (start >= len) return;
    uint end = min(start + chunk_size, len);
    uint count = end - start;

    // Accumulate all points into registers so `acc` is read and written
    // once per element instead of once per (element, point).
    F sums[32];
    for (uint k = 0; k < count; k++) {
        sums[k] = zero_f();
    }
    for (uint j = 0; j < num_points; j++) {
        F point = load_f(points, j);
        // Fold the scalar into the running power so the inner loop needs a
        // single multiplication per element.
        F power = mul_f(load_f(scalars, j), pow_f(load_f(point_steps, j), gid));
        for (uint k = 0; k < count; k++) {
            sums[k] = add_f(sums[k], power);
            power = mul_f(power, point);
        }
    }
    for (uint i = start, k = 0; i < end; i++, k++) {
        store_f(acc, i, add_f(load_f(acc, i), sums[k]));
    }
}

kernel void bn254_geometric_accumulate_point_blocks(
    device ulong *partials [[buffer(0)]],
    device const ulong *points [[buffer(1)]],
    device const ulong *point_steps [[buffer(2)]],
    device const ulong *scalars [[buffer(3)]],
    constant uint &len [[buffer(4)]],
    constant uint &num_points [[buffer(5)]],
    constant uint &chunk_size [[buffer(6)]],
    constant uint &point_block_size [[buffer(7)]],
    constant uint &point_blocks [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint chunk = gid / point_blocks;
    uint point_block = gid - chunk * point_blocks;
    uint start = chunk * chunk_size;
    if (start >= len) return;
    uint end = min(start + chunk_size, len);
    uint point_start = point_block * point_block_size;
    if (point_start >= num_points) return;
    uint point_end = min(point_start + point_block_size, num_points);

    F sums[32];
    for (uint k = 0; k < chunk_size; k++) {
        sums[k] = zero_f();
    }

    for (uint j = point_start; j < point_end; j++) {
        F point = load_f(points, j);
        F power = mul_f(load_f(scalars, j), pow_f(load_f(point_steps, j), chunk));
        for (uint i = start, k = 0; i < end; i++, k++) {
            sums[k] = add_f(sums[k], power);
            power = mul_f(power, point);
        }
    }

    uint partial_offset = point_block * len;
    for (uint i = start, k = 0; i < end; i++, k++) {
        store_f(partials, partial_offset + i, sums[k]);
    }
}

kernel void bn254_geometric_accumulate_point_blocks_range(
    device ulong *partials [[buffer(0)]],
    device const ulong *points [[buffer(1)]],
    device const ulong *point_steps [[buffer(2)]],
    device const ulong *scalars [[buffer(3)]],
    constant uint &len [[buffer(4)]],
    constant uint &num_points [[buffer(5)]],
    constant uint &chunk_size [[buffer(6)]],
    constant uint &point_block_size [[buffer(7)]],
    constant uint &point_block_offset [[buffer(8)]],
    constant uint &batch_point_blocks [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint chunk = gid / batch_point_blocks;
    uint local_point_block = gid - chunk * batch_point_blocks;
    uint global_point_block = point_block_offset + local_point_block;
    uint start = chunk * chunk_size;
    if (start >= len) return;
    uint end = min(start + chunk_size, len);
    uint point_start = global_point_block * point_block_size;
    if (point_start >= num_points) return;
    uint point_end = min(point_start + point_block_size, num_points);

    F sums[32];
    for (uint k = 0; k < chunk_size; k++) {
        sums[k] = zero_f();
    }

    for (uint j = point_start; j < point_end; j++) {
        F point = load_f(points, j);
        F power = mul_f(load_f(scalars, j), pow_f(load_f(point_steps, j), chunk));
        for (uint i = start, k = 0; i < end; i++, k++) {
            sums[k] = add_f(sums[k], power);
            power = mul_f(power, point);
        }
    }

    uint partial_offset = local_point_block * len;
    for (uint i = start, k = 0; i < end; i++, k++) {
        store_f(partials, partial_offset + i, sums[k]);
    }
}

kernel void bn254_geometric_accumulate_reduce_point_blocks(
    device ulong *acc [[buffer(0)]],
    device const ulong *partials [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    constant uint &point_blocks [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    F value = load_f(acc, gid);
    for (uint block = 0; block < point_blocks; block++) {
        value = add_f(value, load_f(partials, block * len + gid));
    }
    store_f(acc, gid, value);
}

kernel void bn254_univariate_evaluate(
    device const ulong *coeffs [[buffer(0)]],
    device const ulong *point_buf [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    if (len == 0) {
        store_f(out, 0, zero_f());
        return;
    }
    F point = load_f(point_buf, 0);
    F acc = load_f(coeffs, len - 1);
    for (uint i = len - 1; i > 0; i--) {
        acc = add_f(mul_f(acc, point), load_f(coeffs, i - 1));
    }
    store_f(out, 0, acc);
}

kernel void bn254_univariate_eval_chunks(
    device const ulong *coeffs [[buffer(0)]],
    device const ulong *point_buf [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &chunk_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * chunk_size;
    if (start >= len) return;
    uint end = min(start + chunk_size, len);
    F point = load_f(point_buf, 0);
    F power = pow_f(point, start);
    F acc = zero_f();
    for (uint i = start; i < end; i++) {
        acc = add_f(acc, mul_f(load_f(coeffs, i), power));
        power = mul_f(power, point);
    }
    store_f(out, gid, acc);
}

kernel void bn254_interleaved_rs_encode(
    device const ulong *coeffs [[buffer(0)]],
    device const ulong *points [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &num_messages [[buffer(3)]],
    constant uint &coeff_len [[buffer(4)]],
    constant uint &total [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint row = gid / num_messages;
    uint message = gid - row * num_messages;
    uint base = message * coeff_len;
    F point = load_f(points, row);
    F acc = load_f(coeffs, base + coeff_len - 1);
    for (uint i = coeff_len - 1; i > 0; i--) {
        acc = add_f(mul_f(acc, point), load_f(coeffs, base + i - 1));
    }
    store_f(out, gid, acc);
}

kernel void bn254_interleaved_rs_encode_single_vector(
    device const ulong *vector [[buffer(0)]],
    device const ulong *points [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &interleaving_depth [[buffer(3)]],
    constant uint &message_length [[buffer(4)]],
    constant uint &total [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint row = gid / interleaving_depth;
    uint message = gid - row * interleaving_depth;
    uint base = message * message_length;
    F point = load_f(points, row);
    F acc = load_f(vector, base + message_length - 1);
    for (uint i = message_length - 1; i > 0; i--) {
        acc = add_f(mul_f(acc, point), load_f(vector, base + i - 1));
    }
    store_f(out, gid, acc);
}

inline uint reverse_bits_width(uint value, uint width) {
    return reverse_bits(value) >> (32u - width);
}

kernel void bn254_pack_single_vector_cosets(
    device const ulong *vector [[buffer(0)]],
    device ulong *out [[buffer(1)]],
    constant PackSingleVectorParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elements) return;
    uint row = gid / params.codeword_length;
    uint col = gid - row * params.codeword_length;
    if (col < params.message_length) {
        store_f(out, gid, load_f(vector, row * params.message_length + col));
    } else {
        store_f(out, gid, zero_f());
    }
}

kernel void bn254_replicate_first_coset(
    device ulong *buffer [[buffer(0)]],
    constant ReplicateCosetsParams &params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.trailing_elements) return;
    uint repeats_per_row = params.row_len - params.coset_size;
    uint row = gid / repeats_per_row;
    uint within = gid - row * repeats_per_row;
    uint dst = row * params.row_len + params.coset_size + within;
    uint src = row * params.row_len + (within % params.coset_size);
    store_f(buffer, dst, load_f(buffer, src));
}

kernel void bn254_bit_reverse_permute_rows_in_place(
    device ulong *values [[buffer(0)]],
    constant BitReverseParams &config [[buffer(1)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= config.total_elements || config.row_len <= 1u) return;
    uint row = index / config.row_len;
    uint within = index - row * config.row_len;
    uint reversed = reverse_bits_width(within, config.log_n);
    if (reversed <= within) return;

    uint row_base = row * config.row_len;
    uint mate = row_base + reversed;
    uint current = row_base + within;
    F tmp = load_f(values, current);
    store_f(values, current, load_f(values, mate));
    store_f(values, mate, tmp);
}

kernel void bn254_radix2_ntt_stage_rows_in_place(
    device ulong *values [[buffer(0)]],
    device const ulong *twiddles [[buffer(1)]],
    constant StageConfig &config [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint butterflies_per_row = config.row_len >> 1u;
    uint row = index / butterflies_per_row;
    uint local = index - row * butterflies_per_row;
    uint half_m = config.half_m;
    uint pair_in_group = local % half_m;
    uint group = local / half_m;
    uint row_base = row * config.row_len;
    uint base = row_base + group * (half_m << 1u) + pair_in_group;
    uint mate = base + half_m;

    F even = load_f(values, base);
    F odd = load_f(values, mate);
    F twiddle = load_f(twiddles, config.twiddle_offset + pair_in_group);
    F t = mul_f(twiddle, odd);

    store_f(values, base, add_f(even, t));
    store_f(values, mate, sub_f(even, t));
}

kernel void bn254_transpose_matrix_reverse_rows(
    device const ulong *input [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant TransposeParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elements) return;
    uint row = gid / params.cols;
    uint col = gid - row * params.cols;
    uint row_bits = 31u - clz(params.cols);
    uint dst_row = reverse_bits_width(col, row_bits);
    uint dst = dst_row * params.rows + row;
    store_f(output, dst, load_f(input, gid));
}

kernel void bn254_transpose_matrix(
    device const ulong *input [[buffer(0)]],
    device ulong *output [[buffer(1)]],
    constant TransposeParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elements) return;
    uint row = gid / params.cols;
    uint col = gid - row * params.cols;
    uint dst = col * params.rows + row;
    store_f(output, dst, load_f(input, gid));
}

kernel void bn254_apply_coset_twiddles(
    device ulong *values [[buffer(0)]],
    device const ulong *root_powers [[buffer(1)]],
    constant ApplyCosetTwiddlesParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elements) return;
    uint within_codeword = gid % params.codeword_length;
    uint coset = within_codeword / params.coset_size;
    uint col = within_codeword - coset * params.coset_size;
    if (coset == 0 || col == 0) return;
    uint root_index = (coset * col) % params.codeword_length;
    store_f(values, gid, mul_f(load_f(values, gid), load_f(root_powers, root_index)));
}

kernel void bn254_encode_field_rows_le(
    device const ulong *input [[buffer(0)]],
    device uchar *output [[buffer(1)]],
    constant FieldBytesParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_elements = params.rows * params.cols;
    if (gid >= total_elements) return;

    F canonical = from_mont(load_f(input, gid));
    uint byte_offset = gid * 32u;
    for (uint limb = 0; limb < 4; ++limb) {
        ulong value = canonical.v[limb];
        for (uint byte = 0; byte < 8; ++byte) {
            output[byte_offset + limb * 8u + byte] = uchar((value >> (byte * 8u)) & 0xffUL);
        }
    }
}

kernel void bn254_read_rows(
    device const ulong *input [[buffer(0)]],
    device const uint *indices [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &num_cols [[buffer(3)]],
    constant uint &total [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total) return;
    uint row = gid / num_cols;
    uint col = gid - row * num_cols;
    uint src = indices[row] * num_cols + col;
    store_f(out, gid, load_f(input, src));
}

kernel void bn254_multilinear_extend(
    device const ulong *values [[buffer(0)]],
    device const ulong *point [[buffer(1)]],
    device ulong *out [[buffer(2)]],
    constant uint &len [[buffer(3)]],
    constant uint &num_vars [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    F acc = zero_f();
    F one = one_f();
    for (uint i = 0; i < len; i++) {
        F weight = one;
        for (uint j = 0; j < num_vars; j++) {
            F r = load_f(point, num_vars - 1 - j);
            if (((i >> j) & 1) != 0) {
                weight = mul_f(weight, r);
            } else {
                weight = mul_f(weight, sub_f(one, r));
            }
        }
        acc = add_f(acc, mul_f(load_f(values, i), weight));
    }
    store_f(out, 0, acc);
}
"#;

const REDUCTION_CHUNK_SIZE: usize = 64;
const SMALL_GEOMETRIC_ACCUMULATE_CHUNK_SIZE: usize = 8;
const LARGE_GEOMETRIC_ACCUMULATE_CHUNK_SIZE: usize = 32;
/// Must match the `sums` register array size in the strided kernel.
const MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE: usize = 32;
const LARGE_GEOMETRIC_ACCUMULATE_THRESHOLD: usize = 1 << 18;
const GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE: usize = 16;
const GEOMETRIC_ACCUMULATE_POINT_BLOCK_THRESHOLD: usize = 1 << 17;
const GEOMETRIC_ACCUMULATE_POINT_BLOCK_MIN_POINTS: usize = 64;
const GEOMETRIC_ACCUMULATE_POINT_BLOCK_BATCH_BYTES: usize = 256 << 20;

#[derive(Clone, Debug)]
struct MetalFieldBuffer {
    limbs: Buffer,
}

#[derive(Clone, Debug)]
struct MetalHashBuffer {
    bytes: Buffer,
}

#[derive(Clone, Debug)]
pub struct MetalBuffer<T> {
    len: usize,
    host_cache: OnceCell<Vec<T>>,
    field: Option<MetalFieldBuffer>,
    hash: Option<MetalHashBuffer>,
    _marker: PhantomData<T>,
}

impl<T: Clone> PartialEq for MetalBuffer<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Clone + Eq> Eq for MetalBuffer<T> {}

impl<T: Clone + PartialOrd> PartialOrd for MetalBuffer<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Clone + Ord> Ord for MetalBuffer<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: Clone + Hash> Hash for MetalBuffer<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<T: Clone + Default> Default for MetalBuffer<T> {
    fn default() -> Self {
        Self {
            len: 0,
            host_cache: OnceCell::new(),
            field: None,
            hash: None,
            _marker: PhantomData,
        }
    }
}

impl<T: Clone + Serialize> Serialize for MetalBuffer<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for MetalBuffer<T>
where
    T: Clone + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = Vec::<T>::deserialize(deserializer)?;
        Ok(Self::from_vec(data))
    }
}

impl<T: Clone> MetalBuffer<T> {
    pub fn warmup() {
        let _ = runtime();
    }

    pub fn from_vec(source: Vec<T>) -> Self {
        let len = source.len();
        let field = maybe_upload_bn254(&source);
        let host_cache = if field.is_some() {
            OnceCell::new()
        } else {
            OnceCell::from(source)
        };
        Self {
            len,
            host_cache,
            field,
            hash: None,
            _marker: PhantomData,
        }
    }

    pub fn from_slice(source: &[T]) -> Self {
        Self::from_vec(Vec::from(source))
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        self.host_cache
            .get_or_init(|| self.download_host_cache())
            .as_slice()
    }

    pub(crate) fn hash_bn254_rows_sha2(&self, num_cols: usize, out: &mut [Digest]) -> bool {
        if type_name::<T>() != type_name::<Field256>() || self.field.is_none() {
            return false;
        }
        assert_eq!(self.len(), num_cols * out.len());
        let message_size = num_cols * size_of::<Field256>();
        let encoded = encode_field_rows_le(
            &self
                .field
                .as_ref()
                .expect("missing Metal field buffer")
                .limbs,
            out.len(),
            num_cols,
        );
        MetalSha2::new().hash_many_buffer(message_size, &encoded, out.len(), out);
        true
    }

    pub(crate) fn commit_bn254_rows_sha2_merkle(
        &self,
        num_cols: usize,
        num_rows: usize,
        layers: usize,
    ) -> Option<MetalBuffer<Digest>> {
        if type_name::<T>() != type_name::<Field256>() || self.field.is_none() {
            return None;
        }
        if num_rows != (1usize << layers) {
            return None;
        }
        assert_eq!(self.len(), num_cols * num_rows);
        let message_size = num_cols * size_of::<Field256>();
        let encoded = encode_field_rows_le(
            &self
                .field
                .as_ref()
                .expect("missing Metal field buffer")
                .limbs,
            num_rows,
            num_cols,
        );
        let sha = MetalSha2::new();
        let nodes = sha.build_merkle_tree_buffer_from_messages_buffer(
            message_size,
            &encoded,
            num_rows,
            layers,
        );
        Some(MetalBuffer::<Digest>::from_digest_buffer(
            nodes,
            (1usize << (layers + 1)) - 1,
        ))
    }
}

impl MetalBuffer<Digest> {
    pub(crate) fn from_digest_buffer(bytes: Buffer, len: usize) -> Self {
        Self {
            len,
            host_cache: OnceCell::new(),
            field: None,
            hash: Some(MetalHashBuffer { bytes }),
            _marker: PhantomData,
        }
    }

    pub(crate) fn read_hash_at(&self, index: usize) -> Option<Digest> {
        self.read_hash_indices(&[index])
            .map(|mut values| values.pop().expect("missing hash"))
    }

    pub(crate) fn read_hash_indices(&self, indices: &[usize]) -> Option<Vec<Digest>> {
        let buffer = self.hash.as_ref()?;
        Some(download_hash_indices(&buffer.bytes, self.len, indices))
    }
}

impl<T: Clone> BufferOps<T> for MetalBuffer<T> {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn read_rows(&self, num_cols: usize, indices: &[usize]) -> Vec<T> {
        if type_name::<T>() == type_name::<Field256>() && self.field.is_some() {
            return read_bn254_rows(
                self.field.as_ref().expect("missing Metal field buffer"),
                num_cols,
                indices,
            )
            .into_iter()
            .map(|value| unsafe { std::mem::transmute_copy(&value) })
            .collect();
        }
        let data = self.as_slice();
        let mut result = Vec::with_capacity(indices.len() * num_cols);
        for i in indices {
            result.extend_from_slice(&data[i * num_cols..(i + 1) * num_cols]);
        }
        result
    }

    fn from_vec(source: Vec<T>) -> Self {
        Self::from_vec(source)
    }

    fn from_slice(source: &[T]) -> Self {
        Self::from_slice(source)
    }
}

impl<F: Field + Clone> FieldOps<F> for MetalBuffer<F> {
    type TargetBuffer<T: Field> = MetalBuffer<T>;

    fn zeros(length: usize) -> Self {
        assert_bn254::<F>();
        // Montgomery zero is all-zero bytes, so a device-side fill suffices.
        Self {
            len: length,
            host_cache: OnceCell::new(),
            field: Some(zeroed_field_buffer(length)),
            hash: None,
            _marker: PhantomData,
        }
    }

    fn random<R>(rng: &mut R, length: usize) -> Self
    where
        R: RngCore + CryptoRng,
        Standard: Distribution<F>,
    {
        assert_bn254::<F>();
        Self::from_vec((0..length).map(|_| rng.gen()).collect())
    }

    fn zero_pad(&mut self) {
        assert_bn254::<F>();
        if !self.is_empty() {
            let mut data = self.as_slice().to_vec();
            data.resize(self.len().next_power_of_two(), F::ZERO);
            *self = Self::from_vec(data);
        }
    }

    fn dot(&self, other: &Self) -> F {
        assert_bn254::<F>();
        assert_eq!(self.len(), other.len());
        let this = self.bn254_buffer();
        let other = other.bn254_buffer();
        field256_to_f::<F>(parallel_dot(&this, &other, self.len()))
    }

    fn fold(&mut self, weight: F) {
        assert_bn254::<F>();
        if self.len() <= 1 {
            return;
        }
        let len = self.len();
        let fold_half = len.next_power_of_two() >> 1;
        let weight = upload_field(&[f_to_field256(weight)]);
        let field = self.bn254_buffer();
        run_in_place(
            "bn254_fold",
            &[&field.limbs, &weight.limbs],
            &[len as u32, fold_half as u32],
            fold_half,
        );
        self.len = fold_half;
        self.invalidate_host_cache();
    }

    fn fold_pair(&mut self, other: &mut Self, weight: F) {
        assert_bn254::<F>();
        assert_eq!(self.len(), other.len());
        if self.len() <= 1 {
            return;
        }
        let len = self.len();
        let fold_half = len.next_power_of_two() >> 1;
        let weight = upload_field(&[f_to_field256(weight)]);
        let this = self.bn254_buffer();
        let other_buffer = other.bn254_buffer();
        run_in_place(
            "bn254_fold_pair",
            &[&this.limbs, &other_buffer.limbs, &weight.limbs],
            &[len as u32, fold_half as u32],
            fold_half,
        );
        self.len = fold_half;
        other.len = fold_half;
        self.invalidate_host_cache();
        other.invalidate_host_cache();
    }

    fn sumcheck_polynomial(&self, other: &Self) -> (F, F) {
        assert_bn254::<F>();
        let len = self.len().min(other.len());
        if len == 0 {
            return (F::ZERO, F::ZERO);
        }
        if len == 1 {
            return (self.as_slice()[0] * other.as_slice()[0], F::ZERO);
        }
        let fold_half = len.next_power_of_two() >> 1;
        let this = self.bn254_buffer();
        let other = other.bn254_buffer();
        let (c0, c2) = parallel_sumcheck(&this, &other, len, fold_half);
        (field256_to_f::<F>(c0), field256_to_f::<F>(c2))
    }

    fn fold_pair_sumcheck_polynomial(&mut self, other: &mut Self, weight: F) -> (F, F) {
        assert_bn254::<F>();
        assert_eq!(self.len(), other.len());
        if self.len() <= 1 {
            return self.sumcheck_polynomial(other);
        }
        let len = self.len();
        let fold_half = len.next_power_of_two() >> 1;
        if fold_half == 1 {
            self.fold_pair(other, weight);
            return self.sumcheck_polynomial(other);
        }
        let weight = upload_field(&[f_to_field256(weight)]);
        let this = self.bn254_buffer();
        let other_buffer = other.bn254_buffer();
        let (c0, c2) = parallel_fold_pair_sumcheck(&this, &other_buffer, &weight, len, fold_half);
        self.len = fold_half;
        other.len = fold_half;
        self.invalidate_host_cache();
        other.invalidate_host_cache();
        (field256_to_f::<F>(c0), field256_to_f::<F>(c2))
    }

    fn accumulate_univariate_evaluations(
        &mut self,
        evaluators: &[UnivariateEvaluation<F>],
        scalars: &[F],
    ) {
        assert_bn254::<F>();
        assert_eq!(evaluators.len(), scalars.len());
        let Some(size) = evaluators.first().map(|e| e.size) else {
            return;
        };
        assert_eq!(self.len(), size);
        for evaluator in evaluators {
            assert_eq!(evaluator.size, size);
        }
        let points = evaluators
            .iter()
            .map(|e| f_to_field256(e.point))
            .collect::<Vec<_>>();
        let scalars = scalars
            .iter()
            .copied()
            .map(f_to_field256)
            .collect::<Vec<_>>();
        let points = upload_field(&points);
        let scalars = upload_field(&scalars);
        let field = self.bn254_buffer();
        let chunk_size = geometric_accumulate_chunk_size(self.len());
        if std::env::var_os("WHIR_METAL_TRACE").is_some() {
            eprintln!(
                "metal geometric shape len={} points={} chunk={} chunks={}",
                self.len(),
                evaluators.len(),
                chunk_size,
                self.len().div_ceil(chunk_size)
            );
        }
        if chunk_size <= 1 {
            run_in_place(
                "bn254_geometric_accumulate",
                &[&field.limbs, &points.limbs, &scalars.limbs],
                &[self.len() as u32, evaluators.len() as u32],
                self.len(),
            );
        } else {
            let point_steps = evaluators
                .iter()
                .map(|e| f_to_field256(e.point.pow([chunk_size as u64])))
                .collect::<Vec<_>>();
            let point_steps = upload_field(&point_steps);
            if should_use_geometric_point_blocks(self.len(), evaluators.len(), chunk_size) {
                parallel_geometric_accumulate_point_blocks(
                    &field,
                    &points,
                    &point_steps,
                    &scalars,
                    self.len(),
                    evaluators.len(),
                    chunk_size,
                );
            } else if should_use_geometric_point_blocks_batched(
                self.len(),
                evaluators.len(),
                chunk_size,
            ) {
                parallel_geometric_accumulate_point_blocks_batched(
                    &field,
                    &points,
                    &point_steps,
                    &scalars,
                    self.len(),
                    evaluators.len(),
                    chunk_size,
                );
            } else {
                run_in_place(
                    "bn254_geometric_accumulate_chunks_strided",
                    &[
                        &field.limbs,
                        &points.limbs,
                        &point_steps.limbs,
                        &scalars.limbs,
                    ],
                    &[
                        self.len() as u32,
                        evaluators.len() as u32,
                        chunk_size as u32,
                    ],
                    self.len().div_ceil(chunk_size),
                );
            }
        }
        self.invalidate_host_cache();
    }

    fn linear_forms_rlc(
        size: usize,
        linear_forms: &mut [Box<dyn LinearForm<F>>],
        rlc_coeffs: &[F],
    ) -> Self {
        assert_bn254::<F>();
        assert_eq!(linear_forms.len(), rlc_coeffs.len());
        let Some((first, rest)) = linear_forms.split_first_mut() else {
            return Self::zeros(size);
        };
        let first = (first.as_mut() as &mut dyn std::any::Any)
            .downcast_mut::<Covector<F>>()
            .expect("MetalBuffer only supports Covector linear forms for BN254 RLC");
        let mut accumulator = Self::from_slice(&first.vector);
        for (coeff, linear_form) in rlc_coeffs[1..].iter().zip(rest) {
            let covector = (linear_form.as_mut() as &mut dyn std::any::Any)
                .downcast_mut::<Covector<F>>()
                .expect("MetalBuffer only supports Covector linear forms for BN254 RLC");
            let vector = Self::from_slice(&covector.vector);
            vector.mixed_scalar_mul_add_to(&Identity::new(), &mut accumulator, *coeff);
        }
        accumulator
    }

    fn mixed_extend<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        _embedding: &M,
        point: &[M::Target],
    ) -> M::Target {
        assert_bn254::<F>();
        assert_bn254::<T>();
        let num_vars = point.len();
        let point = point
            .iter()
            .copied()
            .map(target_to_field256)
            .collect::<Vec<_>>();
        let point = upload_field(&point);
        let this = self.bn254_buffer();
        let out = run_reduce(
            "bn254_multilinear_extend",
            &[&this.limbs, &point.limbs],
            &[self.len() as u32, num_vars as u32],
            1,
        );
        field256_to_target::<M::Target>(out[0])
    }

    fn mixed_dot<M: Embedding<Source = F, Target = T>, T: Field>(
        &self,
        _embedding: &M,
        other: &Self::TargetBuffer<T>,
    ) -> M::Target {
        assert_bn254::<F>();
        assert_bn254::<T>();
        let this = self.bn254_buffer();
        let other = other.bn254_buffer_target();
        let value = field256_to_f::<F>(parallel_dot(&this, &other, self.len()));
        field256_to_target::<M::Target>(f_to_field256(value))
    }

    fn mixed_univariate_evaluate<M: Embedding<Source = F>>(
        &self,
        _embedding: &M,
        point: M::Target,
    ) -> M::Target {
        assert_bn254::<F>();
        let point = target_to_field256(point);
        let point = upload_field(&[point]);
        let this = self.bn254_buffer();
        field256_to_target::<M::Target>(parallel_univariate_evaluate(&this, &point, self.len()))
    }

    fn mixed_linear_combination<M: Embedding<Source = F>>(
        _embedding: &M,
        vectors: &[&Self],
        coeffs: &[M::Target],
    ) -> Self::TargetBuffer<M::Target> {
        assert_bn254::<F>();
        assert_eq!(vectors.len(), coeffs.len());
        let Some((first, vectors)) = vectors.split_first() else {
            return MetalBuffer::from_vec(Vec::new());
        };
        let mut accumulator = MetalBuffer::<M::Target> {
            len: first.len(),
            host_cache: OnceCell::new(),
            field: Some(copy_field_buffer(&first.bn254_buffer(), first.len())),
            hash: None,
            _marker: PhantomData,
        };
        for (coeff, vector) in coeffs[1..].iter().copied().zip(vectors) {
            vector.mixed_scalar_mul_add_to(_embedding, &mut accumulator, coeff);
        }
        accumulator
    }

    fn mixed_scalar_mul_add_to<M: Embedding<Source = F>>(
        &self,
        _embedding: &M,
        accumulator: &mut Self::TargetBuffer<M::Target>,
        weight: M::Target,
    ) {
        assert_bn254::<F>();
        let weight = upload_field(&[target_to_field256(weight)]);
        let vector = self.bn254_buffer();
        let acc = accumulator.bn254_buffer_target();
        run_in_place(
            "bn254_scalar_mul_add",
            &[&acc.limbs, &vector.limbs, &weight.limbs],
            &[self.len() as u32],
            self.len(),
        );
        accumulator.invalidate_host_cache();
    }
}

/// Metal (GPU) Reed-Solomon encoder.
///
/// Wraps the shared [`RsDomain`] core: scalar methods delegate to the domain, and the coset
/// layout used by the GPU encode is taken from [`RsDomain::coset_params`] so it can never
/// drift from [`RsDomain::evaluation_points`]. Unlike the CPU encoder, it does not need the
/// host NTT engine at all.
#[derive(Debug, Clone)]
pub struct MetalRs<F: Field> {
    domain: Arc<RsDomain<F>>,
}

impl<F: Field> MetalRs<F> {
    pub fn new(domain: Arc<RsDomain<F>>) -> Self {
        Self { domain }
    }
}

impl<F: Field> ReedSolomon<F> for MetalRs<F> {
    fn next_order(&self, size: usize) -> Option<usize> {
        self.domain.next_order(size)
    }

    fn generator(&self, codeword_length: usize) -> F {
        self.domain.generator(codeword_length)
    }

    fn evaluation_points(
        &self,
        masked_message_length: usize,
        codeword_length: usize,
        indices: &[usize],
    ) -> Vec<F> {
        self.domain
            .evaluation_points(masked_message_length, codeword_length, indices)
    }

    fn interleaved_encode(
        &self,
        vectors: &[&MetalBuffer<F>],
        masks: &MetalBuffer<F>,
        message_length: usize,
        interleaving_depth: usize,
        codeword_length: usize,
    ) -> MetalBuffer<F> {
        assert_bn254::<F>();
        let num_messages = vectors.len() * interleaving_depth;
        if num_messages == 0 {
            return MetalBuffer::from_vec(Vec::new());
        }
        assert!(masks.len().is_multiple_of(num_messages));
        let mask_length = masks.len() / num_messages;
        if vectors.len() == 1 && mask_length == 0 {
            // Single source of the coset layout: derived from the shared domain rather
            // than recomputed on the device.
            let (coset_size, _num_cosets) =
                self.domain.coset_params(message_length, codeword_length);
            return encode_single_vector_coset_ntt(
                vectors[0],
                message_length,
                interleaving_depth,
                codeword_length,
                coset_size,
            );
        }

        panic!("MetalBuffer BN254 RS encoding supports only one unmasked vector")
    }
}

impl<T: Clone> MetalBuffer<T> {
    fn bn254_buffer(&self) -> MetalFieldBuffer {
        self.field
            .clone()
            .unwrap_or_else(|| upload_field(as_field256_slice(self.as_slice())))
    }

    fn invalidate_host_cache(&mut self) {
        let _ = self.host_cache.take();
    }

    fn download_host_cache(&self) -> Vec<T> {
        if self.field.is_some() && type_name::<T>() == type_name::<Field256>() {
            return download_field(
                &self
                    .field
                    .as_ref()
                    .expect("missing Metal field buffer")
                    .limbs,
                self.len,
            )
            .into_iter()
            .map(|value| unsafe { std::mem::transmute_copy(&value) })
            .collect();
        }
        if self.hash.is_some() && type_name::<T>() == type_name::<Digest>() {
            return download_hash_indices(
                &self.hash.as_ref().expect("missing Metal hash buffer").bytes,
                self.len,
                &(0..self.len).collect::<Vec<_>>(),
            )
            .into_iter()
            .map(|value| unsafe { std::mem::transmute_copy(&value) })
            .collect();
        }
        panic!(
            "MetalBuffer<{}> has no host cache and cannot be materialized",
            type_name::<T>()
        );
    }
}

impl<T: Clone + Field> MetalBuffer<T> {
    fn bn254_buffer_target(&self) -> MetalFieldBuffer {
        self.field
            .clone()
            .unwrap_or_else(|| upload_field(as_field256_slice(self.as_slice())))
    }
}

struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    fold: ComputePipelineState,
    fold_pair: ComputePipelineState,
    scalar_mul_add: ComputePipelineState,
    dot: ComputePipelineState,
    sumcheck: ComputePipelineState,
    dot_chunks: ComputePipelineState,
    sum_chunks: ComputePipelineState,
    sumcheck_reduce_chunks: ComputePipelineState,
    sumcheck_chunks: ComputePipelineState,
    fold_pair_sumcheck_chunks: ComputePipelineState,
    geometric_accumulate: ComputePipelineState,
    geometric_accumulate_chunks: ComputePipelineState,
    geometric_accumulate_chunks_strided: ComputePipelineState,
    geometric_accumulate_point_blocks: ComputePipelineState,
    geometric_accumulate_point_blocks_range: ComputePipelineState,
    geometric_accumulate_reduce_point_blocks: ComputePipelineState,
    univariate_evaluate: ComputePipelineState,
    univariate_eval_chunks: ComputePipelineState,
    interleaved_rs_encode: ComputePipelineState,
    interleaved_rs_encode_single_vector: ComputePipelineState,
    multilinear_extend: ComputePipelineState,
    pack_single_vector_cosets: ComputePipelineState,
    apply_coset_twiddles: ComputePipelineState,
    replicate_first_coset: ComputePipelineState,
    bit_reverse_rows: ComputePipelineState,
    ntt_stage_rows: ComputePipelineState,
    transpose: ComputePipelineState,
    transpose_reverse_rows: ComputePipelineState,
    encode_field_rows_le: ComputePipelineState,
    read_rows: ComputePipelineState,
    ntt_roots: Mutex<HashMap<usize, MetalFieldBuffer>>,
    root_powers: Mutex<HashMap<usize, MetalFieldBuffer>>,
}

fn runtime() -> &'static MetalRuntime {
    static RUNTIME: OnceLock<MetalRuntime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        let device = Device::system_default().expect("Metal device is not available");
        let library = device
            .new_library_with_source(METAL_SOURCE, &CompileOptions::new())
            .expect("failed to compile Metal BN254 kernels");
        let pipeline = |name: &str| {
            let function = library
                .get_function(name, None)
                .unwrap_or_else(|_| panic!("missing Metal kernel {name}"));
            device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap_or_else(|err| panic!("failed to compile Metal kernel {name}: {err}"))
        };
        MetalRuntime {
            queue: device.new_command_queue(),
            fold: pipeline("bn254_fold"),
            fold_pair: pipeline("bn254_fold_pair"),
            scalar_mul_add: pipeline("bn254_scalar_mul_add"),
            dot: pipeline("bn254_dot"),
            sumcheck: pipeline("bn254_sumcheck"),
            dot_chunks: pipeline("bn254_dot_chunks"),
            sum_chunks: pipeline("bn254_sum_chunks"),
            sumcheck_reduce_chunks: pipeline("bn254_sumcheck_reduce_chunks"),
            sumcheck_chunks: pipeline("bn254_sumcheck_chunks"),
            fold_pair_sumcheck_chunks: pipeline("bn254_fold_pair_sumcheck_chunks"),
            geometric_accumulate: pipeline("bn254_geometric_accumulate"),
            geometric_accumulate_chunks: pipeline("bn254_geometric_accumulate_chunks"),
            geometric_accumulate_chunks_strided: pipeline(
                "bn254_geometric_accumulate_chunks_strided",
            ),
            geometric_accumulate_point_blocks: pipeline("bn254_geometric_accumulate_point_blocks"),
            geometric_accumulate_point_blocks_range: pipeline(
                "bn254_geometric_accumulate_point_blocks_range",
            ),
            geometric_accumulate_reduce_point_blocks: pipeline(
                "bn254_geometric_accumulate_reduce_point_blocks",
            ),
            univariate_evaluate: pipeline("bn254_univariate_evaluate"),
            univariate_eval_chunks: pipeline("bn254_univariate_eval_chunks"),
            interleaved_rs_encode: pipeline("bn254_interleaved_rs_encode"),
            interleaved_rs_encode_single_vector: pipeline(
                "bn254_interleaved_rs_encode_single_vector",
            ),
            multilinear_extend: pipeline("bn254_multilinear_extend"),
            pack_single_vector_cosets: pipeline("bn254_pack_single_vector_cosets"),
            apply_coset_twiddles: pipeline("bn254_apply_coset_twiddles"),
            replicate_first_coset: pipeline("bn254_replicate_first_coset"),
            bit_reverse_rows: pipeline("bn254_bit_reverse_permute_rows_in_place"),
            ntt_stage_rows: pipeline("bn254_radix2_ntt_stage_rows_in_place"),
            transpose: pipeline("bn254_transpose_matrix"),
            transpose_reverse_rows: pipeline("bn254_transpose_matrix_reverse_rows"),
            encode_field_rows_le: pipeline("bn254_encode_field_rows_le"),
            read_rows: pipeline("bn254_read_rows"),
            ntt_roots: Mutex::new(HashMap::new()),
            root_powers: Mutex::new(HashMap::new()),
            device,
        }
    })
}

fn pipeline<'a>(rt: &'a MetalRuntime, name: &str) -> &'a ComputePipelineState {
    match name {
        "bn254_fold" => &rt.fold,
        "bn254_fold_pair" => &rt.fold_pair,
        "bn254_scalar_mul_add" => &rt.scalar_mul_add,
        "bn254_dot" => &rt.dot,
        "bn254_sumcheck" => &rt.sumcheck,
        "bn254_dot_chunks" => &rt.dot_chunks,
        "bn254_sum_chunks" => &rt.sum_chunks,
        "bn254_sumcheck_reduce_chunks" => &rt.sumcheck_reduce_chunks,
        "bn254_sumcheck_chunks" => &rt.sumcheck_chunks,
        "bn254_fold_pair_sumcheck_chunks" => &rt.fold_pair_sumcheck_chunks,
        "bn254_geometric_accumulate" => &rt.geometric_accumulate,
        "bn254_geometric_accumulate_chunks" => &rt.geometric_accumulate_chunks,
        "bn254_geometric_accumulate_chunks_strided" => &rt.geometric_accumulate_chunks_strided,
        "bn254_geometric_accumulate_point_blocks" => &rt.geometric_accumulate_point_blocks,
        "bn254_geometric_accumulate_point_blocks_range" => {
            &rt.geometric_accumulate_point_blocks_range
        }
        "bn254_geometric_accumulate_reduce_point_blocks" => {
            &rt.geometric_accumulate_reduce_point_blocks
        }
        "bn254_univariate_evaluate" => &rt.univariate_evaluate,
        "bn254_univariate_eval_chunks" => &rt.univariate_eval_chunks,
        "bn254_interleaved_rs_encode" => &rt.interleaved_rs_encode,
        "bn254_interleaved_rs_encode_single_vector" => &rt.interleaved_rs_encode_single_vector,
        "bn254_multilinear_extend" => &rt.multilinear_extend,
        "bn254_pack_single_vector_cosets" => &rt.pack_single_vector_cosets,
        "bn254_apply_coset_twiddles" => &rt.apply_coset_twiddles,
        "bn254_replicate_first_coset" => &rt.replicate_first_coset,
        "bn254_bit_reverse_permute_rows_in_place" => &rt.bit_reverse_rows,
        "bn254_radix2_ntt_stage_rows_in_place" => &rt.ntt_stage_rows,
        "bn254_transpose_matrix" => &rt.transpose,
        "bn254_transpose_matrix_reverse_rows" => &rt.transpose_reverse_rows,
        "bn254_encode_field_rows_le" => &rt.encode_field_rows_le,
        "bn254_read_rows" => &rt.read_rows,
        _ => panic!("unknown Metal kernel {name}"),
    }
}

fn new_shared_buffer(rt: &MetalRuntime, bytes: u64) -> Buffer {
    metal_profile::record_alloc(bytes);
    let buffer = rt
        .device
        .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    metal_profile::record_device_allocated(rt.device.current_allocated_size());
    buffer
}

fn new_shared_buffer_with_data(rt: &MetalRuntime, data: *const c_void, bytes: u64) -> Buffer {
    let start = Instant::now();
    let buffer = rt
        .device
        .new_buffer_with_data(data, bytes, MTLResourceOptions::StorageModeShared);
    metal_profile::record_alloc(bytes);
    metal_profile::record_upload(bytes, start.elapsed());
    metal_profile::record_device_allocated(rt.device.current_allocated_size());
    buffer
}

fn wait_for_command_named(command: &metal::CommandBufferRef, label: &str) {
    command.commit();
    let start = Instant::now();
    command.wait_until_completed();
    let elapsed = start.elapsed();
    if std::env::var_os("WHIR_METAL_TRACE").is_some() {
        eprintln!(
            "metal command {label} {:.3} ms",
            elapsed.as_secs_f64() * 1_000.0
        );
    }
    metal_profile::record_command_wait(elapsed);
}

fn wait_for_blit(command: &metal::CommandBufferRef, bytes: u64) {
    command.commit();
    let start = Instant::now();
    command.wait_until_completed();
    metal_profile::record_blit(bytes, start.elapsed());
}

#[repr(C)]
#[derive(Clone, Copy)]
struct BitReverseParams {
    row_len: u32,
    log_n: u32,
    total_elements: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NttStageParams {
    row_len: u32,
    half_m: u32,
    twiddle_offset: u32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TransposeParams {
    rows: u32,
    cols: u32,
    total_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ReplicateCosetsParams {
    row_len: u32,
    coset_size: u32,
    trailing_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PackSingleVectorParams {
    row_count: u32,
    message_length: u32,
    codeword_length: u32,
    coset_size: u32,
    total_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FieldBytesParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ApplyCosetTwiddlesParams {
    row_count: u32,
    num_cosets: u32,
    coset_size: u32,
    codeword_length: u32,
    total_elements: u32,
}

fn parallel_dot(a: &MetalFieldBuffer, b: &MetalFieldBuffer, len: usize) -> Field256 {
    if len == 0 {
        return Field256::ZERO;
    }
    let partial_count = len.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel(
        command,
        pipeline(rt, "bn254_dot_chunks"),
        &[&a.limbs, &b.limbs, &partials.limbs],
        &[len as u32, REDUCTION_CHUNK_SIZE as u32],
        partial_count,
    );
    let (result, offset) = encode_field_reduction(command, partials, partial_count, 0);
    wait_for_command_named(command, "bn254_dot_chunks");
    download_field_at(&result.limbs, offset)
}

fn parallel_sumcheck(
    a: &MetalFieldBuffer,
    b: &MetalFieldBuffer,
    len: usize,
    fold_half: usize,
) -> (Field256, Field256) {
    let partial_count = fold_half.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * 2 * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel(
        command,
        pipeline(rt, "bn254_sumcheck_chunks"),
        &[&a.limbs, &b.limbs, &partials.limbs],
        &[
            len as u32,
            fold_half as u32,
            REDUCTION_CHUNK_SIZE as u32,
            partial_count as u32,
        ],
        partial_count,
    );
    let result = encode_sumcheck_reduction(command, partials, partial_count);
    wait_for_command_named(command, "bn254_sumcheck_chunks");
    download_sumcheck_pair(&result)
}

fn parallel_fold_pair_sumcheck(
    a: &MetalFieldBuffer,
    b: &MetalFieldBuffer,
    weight: &MetalFieldBuffer,
    len: usize,
    fold_half: usize,
) -> (Field256, Field256) {
    let sum_half = fold_half.next_power_of_two() >> 1;
    debug_assert!(sum_half > 0);
    let partial_count = sum_half.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * 2 * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel(
        command,
        pipeline(rt, "bn254_fold_pair_sumcheck_chunks"),
        &[&a.limbs, &b.limbs, &weight.limbs, &partials.limbs],
        &[
            len as u32,
            fold_half as u32,
            sum_half as u32,
            REDUCTION_CHUNK_SIZE as u32,
            partial_count as u32,
        ],
        partial_count,
    );
    let result = encode_sumcheck_reduction(command, partials, partial_count);
    wait_for_command_named(command, "bn254_fold_pair_sumcheck_chunks");
    download_sumcheck_pair(&result)
}

fn parallel_univariate_evaluate(
    coeffs: &MetalFieldBuffer,
    point: &MetalFieldBuffer,
    len: usize,
) -> Field256 {
    if len == 0 {
        return Field256::ZERO;
    }
    let partial_count = len.div_ceil(REDUCTION_CHUNK_SIZE);
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_count * size_of::<Field256>()) as u64),
    };
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel(
        command,
        pipeline(rt, "bn254_univariate_eval_chunks"),
        &[&coeffs.limbs, &point.limbs, &partials.limbs],
        &[len as u32, REDUCTION_CHUNK_SIZE as u32],
        partial_count,
    );
    let (result, offset) = encode_field_reduction(command, partials, partial_count, 0);
    wait_for_command_named(command, "bn254_univariate_eval_chunks");
    download_field_at(&result.limbs, offset)
}

fn parallel_geometric_accumulate_point_blocks(
    acc: &MetalFieldBuffer,
    points: &MetalFieldBuffer,
    point_steps: &MetalFieldBuffer,
    scalars: &MetalFieldBuffer,
    len: usize,
    num_points: usize,
    chunk_size: usize,
) {
    assert!(chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE);
    let point_blocks = num_points.div_ceil(GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE);
    let chunks = len.div_ceil(chunk_size);
    let partial_len = point_blocks
        .checked_mul(len)
        .expect("Metal geometric partial size overflow");
    let rt = runtime();
    let partials = MetalFieldBuffer {
        limbs: new_shared_buffer(rt, (partial_len * size_of::<Field256>()) as u64),
    };
    run_in_place(
        "bn254_geometric_accumulate_point_blocks",
        &[
            &partials.limbs,
            &points.limbs,
            &point_steps.limbs,
            &scalars.limbs,
        ],
        &[
            len as u32,
            num_points as u32,
            chunk_size as u32,
            GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE as u32,
            point_blocks as u32,
        ],
        chunks * point_blocks,
    );
    run_in_place(
        "bn254_geometric_accumulate_reduce_point_blocks",
        &[&acc.limbs, &partials.limbs],
        &[len as u32, point_blocks as u32],
        len,
    );
}

fn parallel_geometric_accumulate_point_blocks_batched(
    acc: &MetalFieldBuffer,
    points: &MetalFieldBuffer,
    point_steps: &MetalFieldBuffer,
    scalars: &MetalFieldBuffer,
    len: usize,
    num_points: usize,
    chunk_size: usize,
) {
    assert!(chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE);
    let point_blocks = num_points.div_ceil(GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE);
    let chunks = len.div_ceil(chunk_size);
    let bytes_per_point_block = len
        .checked_mul(size_of::<Field256>())
        .expect("Metal geometric partial size overflow");
    let default_batch_blocks =
        (GEOMETRIC_ACCUMULATE_POINT_BLOCK_BATCH_BYTES / bytes_per_point_block).max(1);
    let batch_blocks = std::env::var("WHIR_METAL_GEOM_POINT_BLOCK_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default_batch_blocks)
        .min(point_blocks);
    let rt = runtime();
    for point_block_offset in (0..point_blocks).step_by(batch_blocks) {
        let current_batch = batch_blocks.min(point_blocks - point_block_offset);
        let partial_len = current_batch
            .checked_mul(len)
            .expect("Metal geometric partial size overflow");
        let partials = MetalFieldBuffer {
            limbs: new_shared_buffer(rt, (partial_len * size_of::<Field256>()) as u64),
        };
        run_in_place(
            "bn254_geometric_accumulate_point_blocks_range",
            &[
                &partials.limbs,
                &points.limbs,
                &point_steps.limbs,
                &scalars.limbs,
            ],
            &[
                len as u32,
                num_points as u32,
                chunk_size as u32,
                GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE as u32,
                point_block_offset as u32,
                current_batch as u32,
            ],
            chunks * current_batch,
        );
        run_in_place(
            "bn254_geometric_accumulate_reduce_point_blocks",
            &[&acc.limbs, &partials.limbs],
            &[len as u32, current_batch as u32],
            len,
        );
    }
}

fn geometric_accumulate_chunk_size(len: usize) -> usize {
    std::env::var("WHIR_METAL_GEOM_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(if len >= LARGE_GEOMETRIC_ACCUMULATE_THRESHOLD {
            LARGE_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        } else {
            SMALL_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        })
        .min(MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE)
}

fn should_use_geometric_point_blocks(len: usize, num_points: usize, chunk_size: usize) -> bool {
    chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        && num_points >= GEOMETRIC_ACCUMULATE_POINT_BLOCK_MIN_POINTS
        && len <= GEOMETRIC_ACCUMULATE_POINT_BLOCK_THRESHOLD
}

fn should_use_geometric_point_blocks_batched(
    len: usize,
    num_points: usize,
    chunk_size: usize,
) -> bool {
    chunk_size <= MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE
        && num_points >= GEOMETRIC_ACCUMULATE_POINT_BLOCK_MIN_POINTS
        && len > GEOMETRIC_ACCUMULATE_POINT_BLOCK_THRESHOLD
}

/// Encodes all tree-reduction levels into `command` and returns the buffer
/// and offset holding the final scalar. Runs with a single command wait.
fn encode_field_reduction(
    command: &metal::CommandBufferRef,
    input: MetalFieldBuffer,
    len: usize,
    offset: usize,
) -> (MetalFieldBuffer, usize) {
    let rt = runtime();
    let mut current = input;
    let mut current_len = len;
    let mut current_offset = offset;
    while current_len > 1 {
        let next_len = current_len.div_ceil(REDUCTION_CHUNK_SIZE);
        let next = MetalFieldBuffer {
            limbs: new_shared_buffer(rt, (next_len * size_of::<Field256>()) as u64),
        };
        encode_u32_kernel(
            command,
            pipeline(rt, "bn254_sum_chunks"),
            &[&current.limbs, &next.limbs],
            &[
                current_len as u32,
                current_offset as u32,
                REDUCTION_CHUNK_SIZE as u32,
            ],
            next_len,
        );
        current = next;
        current_len = next_len;
        current_offset = 0;
    }
    (current, current_offset)
}

/// Encodes all (c0, c2) tree-reduction levels into `command` and returns the
/// buffer holding the final pair. Runs with a single command wait.
fn encode_sumcheck_reduction(
    command: &metal::CommandBufferRef,
    input: MetalFieldBuffer,
    len: usize,
) -> MetalFieldBuffer {
    let rt = runtime();
    let mut current = input;
    let mut current_len = len;
    while current_len > 1 {
        let next_len = current_len.div_ceil(REDUCTION_CHUNK_SIZE);
        let next = MetalFieldBuffer {
            limbs: new_shared_buffer(rt, (next_len * 2 * size_of::<Field256>()) as u64),
        };
        encode_u32_kernel(
            command,
            pipeline(rt, "bn254_sumcheck_reduce_chunks"),
            &[&current.limbs, &next.limbs],
            &[
                current_len as u32,
                REDUCTION_CHUNK_SIZE as u32,
                next_len as u32,
            ],
            next_len,
        );
        current = next;
        current_len = next_len;
    }
    current
}

fn download_sumcheck_pair(buffer: &MetalFieldBuffer) -> (Field256, Field256) {
    let values = download_field(&buffer.limbs, 2);
    (values[0], values[1])
}

fn encode_single_vector_coset_ntt<F: Field>(
    vector: &MetalBuffer<F>,
    message_length: usize,
    interleaving_depth: usize,
    codeword_length: usize,
    coset_size: usize,
) -> MetalBuffer<F> {
    assert!(codeword_length.is_power_of_two());
    assert!(Field256::get_root_of_unity(codeword_length as u64).is_some());
    assert_eq!(vector.len(), message_length * interleaving_depth);

    assert!(codeword_length.is_multiple_of(coset_size));
    let num_cosets = codeword_length / coset_size;
    let total_elements = interleaving_depth
        .checked_mul(codeword_length)
        .expect("Metal RS encode size overflow");
    assert!(total_elements <= u32::MAX as usize);

    let rt = runtime();
    let source = vector.bn254_buffer();
    let current = new_shared_buffer(rt, (total_elements * 4 * size_of::<u64>()) as u64);
    let transposed = new_shared_buffer(rt, (total_elements * 4 * size_of::<u64>()) as u64);
    let codeword_root_powers = root_powers_buffer(codeword_length);
    let coset_roots = roots_buffer(coset_size);

    let command = rt.queue.new_command_buffer();

    encode_kernel(
        &command,
        &rt.pack_single_vector_cosets,
        &[&source.limbs, &current],
        &PackSingleVectorParams {
            row_count: interleaving_depth as u32,
            message_length: message_length as u32,
            codeword_length: codeword_length as u32,
            coset_size: coset_size as u32,
            total_elements: total_elements as u32,
        },
        total_elements,
    );

    let trailing_elements = interleaving_depth.saturating_mul(codeword_length - coset_size);
    if trailing_elements != 0 {
        encode_kernel(
            &command,
            &rt.replicate_first_coset,
            &[&current],
            &ReplicateCosetsParams {
                row_len: codeword_length as u32,
                coset_size: coset_size as u32,
                trailing_elements: trailing_elements as u32,
            },
            trailing_elements,
        );
    }

    encode_kernel(
        &command,
        &rt.apply_coset_twiddles,
        &[&current, &codeword_root_powers.limbs],
        &ApplyCosetTwiddlesParams {
            row_count: interleaving_depth as u32,
            num_cosets: num_cosets as u32,
            coset_size: coset_size as u32,
            codeword_length: codeword_length as u32,
            total_elements: total_elements as u32,
        },
        total_elements,
    );

    let stage_count = coset_size.trailing_zeros() as usize;
    encode_kernel(
        &command,
        &rt.bit_reverse_rows,
        &[&current],
        &BitReverseParams {
            row_len: coset_size as u32,
            log_n: stage_count as u32,
            total_elements: total_elements as u32,
            _pad0: 0,
        },
        total_elements,
    );

    let total_butterflies = total_elements / 2;
    let mut twiddle_offset = 0usize;
    for stage in 0..stage_count {
        let half_m = 1usize << stage;
        encode_kernel(
            &command,
            &rt.ntt_stage_rows,
            &[&current, &coset_roots.limbs],
            &NttStageParams {
                row_len: coset_size as u32,
                half_m: half_m as u32,
                twiddle_offset: twiddle_offset as u32,
                _pad0: 0,
            },
            total_butterflies,
        );
        twiddle_offset += 1usize << stage;
    }

    encode_kernel(
        &command,
        &rt.transpose,
        &[&current, &transposed],
        &TransposeParams {
            rows: interleaving_depth as u32,
            cols: codeword_length as u32,
            total_elements: total_elements as u32,
        },
        total_elements,
    );

    wait_for_command_named(&command, "bn254_rs_encode");

    MetalBuffer {
        len: total_elements,
        host_cache: OnceCell::new(),
        field: Some(MetalFieldBuffer { limbs: transposed }),
        hash: None,
        _marker: PhantomData,
    }
}

fn encode_kernel<P: Copy>(
    command: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    params: &P,
    threads: usize,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), 0);
    }
    encoder.set_bytes(
        buffers.len() as u64,
        size_of::<P>() as u64,
        (params as *const P).cast::<c_void>(),
    );
    dispatch(&encoder, pipeline, threads.max(1));
    encoder.end_encoding();
}

fn roots_buffer(codeword_length: usize) -> MetalFieldBuffer {
    let rt = runtime();
    if let Some(buffer) = rt.ntt_roots.lock().unwrap().get(&codeword_length).cloned() {
        return buffer;
    }
    let root = Field256::get_root_of_unity(codeword_length as u64)
        .expect("BN254 root of unity unavailable for Metal NTT");
    let stage_count = codeword_length.trailing_zeros() as usize;
    let mut roots = Vec::with_capacity(codeword_length.saturating_sub(1));
    for stage in 0..stage_count {
        let stage_size = 1usize << (stage + 1);
        let half_stage = stage_size >> 1;
        let stage_root = root.pow([(codeword_length / stage_size) as u64]);
        let mut current = Field256::ONE;
        for _ in 0..half_stage {
            roots.push(current);
            current *= stage_root;
        }
    }
    let buffer = upload_field(&roots);
    rt.ntt_roots
        .lock()
        .unwrap()
        .insert(codeword_length, buffer.clone());
    buffer
}

fn root_powers_buffer(codeword_length: usize) -> MetalFieldBuffer {
    let rt = runtime();
    if let Some(buffer) = rt
        .root_powers
        .lock()
        .unwrap()
        .get(&codeword_length)
        .cloned()
    {
        return buffer;
    }
    let root = Field256::get_root_of_unity(codeword_length as u64)
        .expect("BN254 root of unity unavailable for Metal RS twiddles");
    let mut powers = Vec::with_capacity(codeword_length);
    let mut current = Field256::ONE;
    for _ in 0..codeword_length {
        powers.push(current);
        current *= root;
    }
    let buffer = upload_field(&powers);
    rt.root_powers
        .lock()
        .unwrap()
        .insert(codeword_length, buffer.clone());
    buffer
}

fn encode_field_rows_le(input: &Buffer, rows: usize, cols: usize) -> Buffer {
    assert!(rows <= u32::MAX as usize);
    assert!(cols <= u32::MAX as usize);
    let rt = runtime();
    let output = new_shared_buffer(rt, (rows * cols * size_of::<Field256>()) as u64);
    let command = rt.queue.new_command_buffer();
    encode_kernel(
        &command,
        &rt.encode_field_rows_le,
        &[input, &output],
        &FieldBytesParams {
            rows: rows as u32,
            cols: cols as u32,
        },
        rows * cols,
    );
    wait_for_command_named(&command, "bn254_encode_field_rows_le");
    output
}

fn read_bn254_rows(source: &MetalFieldBuffer, num_cols: usize, indices: &[usize]) -> Vec<Field256> {
    if indices.is_empty() || num_cols == 0 {
        return Vec::new();
    }
    assert!(num_cols <= u32::MAX as usize);
    assert!(indices.iter().all(|&index| index <= u32::MAX as usize));
    let total = indices
        .len()
        .checked_mul(num_cols)
        .expect("Metal read_rows size overflow");
    assert!(total <= u32::MAX as usize);

    let rt = runtime();
    let indices = indices
        .iter()
        .copied()
        .map(|index| index as u32)
        .collect::<Vec<_>>();
    let indices_buffer = new_shared_buffer_with_data(
        rt,
        indices.as_ptr().cast(),
        (indices.len() * size_of::<u32>()) as u64,
    );
    let out = new_shared_buffer(rt, (total * size_of::<Field256>()) as u64);
    run_in_place(
        "bn254_read_rows",
        &[&source.limbs, &indices_buffer, &out],
        &[num_cols as u32, total as u32],
        total,
    );
    download_field(&out, total)
}

fn upload_field(values: &[Field256]) -> MetalFieldBuffer {
    let rt = runtime();
    let buffer = if values.is_empty() {
        new_shared_buffer(rt, 0)
    } else {
        // Field256 is 4 contiguous u64 limbs; upload directly without
        // flattening into an intermediate Vec.
        new_shared_buffer_with_data(
            rt,
            values.as_ptr().cast(),
            std::mem::size_of_val(values) as u64,
        )
    };
    MetalFieldBuffer { limbs: buffer }
}

fn zeroed_field_buffer(len: usize) -> MetalFieldBuffer {
    let rt = runtime();
    let bytes = (len * size_of::<Field256>()) as u64;
    let buffer = new_shared_buffer(rt, bytes);
    if bytes > 0 {
        let command = rt.queue.new_command_buffer();
        let blit = command.new_blit_command_encoder();
        blit.fill_buffer(&buffer, metal::NSRange::new(0, bytes), 0);
        blit.end_encoding();
        wait_for_blit(command, bytes);
    }
    MetalFieldBuffer { limbs: buffer }
}

fn maybe_upload_bn254<T: Clone>(values: &[T]) -> Option<MetalFieldBuffer> {
    (type_name::<T>() == type_name::<Field256>()).then(|| upload_field(as_field256_slice(values)))
}

fn copy_field_buffer(source: &MetalFieldBuffer, len: usize) -> MetalFieldBuffer {
    let rt = runtime();
    let byte_len = (len * 4 * size_of::<u64>()) as u64;
    let target = new_shared_buffer(rt, byte_len);
    let command = rt.queue.new_command_buffer();
    let blit = command.new_blit_command_encoder();
    blit.copy_from_buffer(&source.limbs, 0, &target, 0, byte_len);
    blit.end_encoding();
    wait_for_blit(&command, byte_len);
    MetalFieldBuffer { limbs: target }
}

/// Encodes a kernel dispatch with `u32` constants bound as inline bytes
/// (no per-constant buffer allocations).
fn encode_u32_kernel(
    command: &metal::CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    constants: &[u32],
    threads: usize,
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    let mut index = 0;
    for buffer in buffers {
        encoder.set_buffer(index, Some(buffer), 0);
        index += 1;
    }
    for constant in constants {
        encoder.set_bytes(
            index,
            size_of::<u32>() as u64,
            (constant as *const u32).cast(),
        );
        index += 1;
    }
    dispatch(&encoder, pipeline, threads.max(1));
    encoder.end_encoding();
}

fn run_in_place(name: &str, buffers: &[&Buffer], constants: &[u32], threads: usize) {
    let rt = runtime();
    let command = rt.queue.new_command_buffer();
    encode_u32_kernel(command, pipeline(rt, name), buffers, constants, threads);
    wait_for_command_named(command, name);
}

fn run_reduce(name: &str, buffers: &[&Buffer], constants: &[u32], out_len: usize) -> Vec<Field256> {
    let rt = runtime();
    let out = new_shared_buffer(rt, (out_len * 4 * size_of::<u64>()) as u64);
    let mut all = buffers.to_vec();
    all.push(&out);
    run_in_place(name, &all, constants, 1);
    download_field(&out, out_len)
}

fn dispatch(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    threads: usize,
) {
    // Use full threadgroups (capped at 256) instead of a single execution
    // width; the pipeline limit already accounts for register pressure.
    let width = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
    let group_width = width.min(threads as u64).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: threads as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: group_width,
            height: 1,
            depth: 1,
        },
    );
}

fn download_field(buffer: &Buffer, len: usize) -> Vec<Field256> {
    if len == 0 {
        return Vec::new();
    }
    let start = Instant::now();
    let limbs = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u64>(), len * 4) };
    let result = limbs
        .chunks_exact(4)
        .map(|chunk| {
            Fp::<MontBackend<BN254Config, 4>, 4>(
                BigInt([chunk[0], chunk[1], chunk[2], chunk[3]]),
                PhantomData,
            )
        })
        .collect();
    metal_profile::record_readback((len * size_of::<Field256>()) as u64, start.elapsed());
    result
}

fn download_field_at(buffer: &Buffer, index: usize) -> Field256 {
    let start = Instant::now();
    let limbs =
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u64>().add(index * 4), 4) };
    let result = Fp::<MontBackend<BN254Config, 4>, 4>(
        BigInt([limbs[0], limbs[1], limbs[2], limbs[3]]),
        PhantomData,
    );
    metal_profile::record_readback(size_of::<Field256>() as u64, start.elapsed());
    result
}

fn download_hash_indices(buffer: &Buffer, len: usize, indices: &[usize]) -> Vec<Digest> {
    let start = Instant::now();
    let bytes = unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), len * 32) };
    let mut result = Vec::with_capacity(indices.len());
    for &index in indices {
        assert!(index < len, "Metal hash index out of bounds");
        let mut hash = Digest::default();
        hash.as_mut_bytes()
            .copy_from_slice(&bytes[index * 32..(index + 1) * 32]);
        result.push(hash);
    }
    metal_profile::record_readback(
        (indices.len() * size_of::<Digest>()) as u64,
        start.elapsed(),
    );
    result
}

fn assert_bn254<F>() {
    assert_eq!(
        type_name::<F>(),
        type_name::<Field256>(),
        "MetalBuffer only supports BN254 Field256 field operations"
    );
}

fn f_to_field256<F: Field>(value: F) -> Field256 {
    assert_bn254::<F>();
    unsafe { std::mem::transmute_copy(&value) }
}

fn field256_to_f<F: Field>(value: Field256) -> F {
    assert_bn254::<F>();
    unsafe { std::mem::transmute_copy(&value) }
}

fn target_to_field256<T: Field>(value: T) -> Field256 {
    assert_bn254::<T>();
    unsafe { std::mem::transmute_copy(&value) }
}

fn field256_to_target<T: Field>(value: Field256) -> T {
    assert_bn254::<T>();
    unsafe { std::mem::transmute_copy(&value) }
}

fn as_field256_slice<T: Clone>(values: &[T]) -> &[Field256] {
    assert_eq!(
        type_name::<T>(),
        type_name::<Field256>(),
        "MetalBuffer only supports BN254 Field256 buffers"
    );
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{buffer::CpuBuffer, ntt::NttEngine};

    fn values(len: usize, offset: u64) -> Vec<Field256> {
        (0..len)
            .map(|i| Field256::from(i as u64 + offset))
            .collect()
    }

    #[test]
    fn metal_bn254_dot_matches_cpu() {
        let a = values(33, 1);
        let b = values(33, 9);
        let cpu_a = CpuBuffer::from_slice(&a);
        let cpu_b = CpuBuffer::from_slice(&b);
        let gpu_a = MetalBuffer::from_slice(&a);
        let gpu_b = MetalBuffer::from_slice(&b);
        assert_eq!(gpu_a.dot(&gpu_b), cpu_a.dot(&cpu_b));
    }

    #[test]
    fn metal_bn254_fold_matches_cpu() {
        let mut cpu = CpuBuffer::from_vec(values(31, 2));
        let mut gpu = MetalBuffer::from_vec(values(31, 2));
        let weight = Field256::from(42);
        cpu.fold(weight);
        gpu.fold(weight);
        assert_eq!(gpu.as_slice(), cpu.as_slice());
    }

    #[test]
    fn metal_bn254_sumcheck_matches_cpu() {
        for len in [1, 2, 27, 64, 65] {
            let a = values(len, 3);
            let b = values(len, 11);
            let cpu_a = CpuBuffer::from_slice(&a);
            let cpu_b = CpuBuffer::from_slice(&b);
            let gpu_a = MetalBuffer::from_slice(&a);
            let gpu_b = MetalBuffer::from_slice(&b);
            assert_eq!(
                gpu_a.sumcheck_polynomial(&gpu_b),
                cpu_a.sumcheck_polynomial(&cpu_b)
            );
        }
    }

    #[test]
    fn metal_bn254_fold_pair_sumcheck_matches_cpu() {
        for len in [2, 3, 27, 64, 65] {
            let mut cpu_a = CpuBuffer::from_vec(values(len, 3));
            let mut cpu_b = CpuBuffer::from_vec(values(len, 11));
            let mut gpu_a = MetalBuffer::from_vec(values(len, 3));
            let mut gpu_b = MetalBuffer::from_vec(values(len, 11));
            let weight = Field256::from(42);
            let cpu_result = cpu_a.fold_pair_sumcheck_polynomial(&mut cpu_b, weight);
            let gpu_result = gpu_a.fold_pair_sumcheck_polynomial(&mut gpu_b, weight);
            assert_eq!(gpu_result, cpu_result);
            assert_eq!(gpu_a.as_slice(), cpu_a.as_slice());
            assert_eq!(gpu_b.as_slice(), cpu_b.as_slice());
        }
    }

    #[test]
    fn metal_bn254_scalar_mul_add_matches_cpu() {
        let mut cpu = CpuBuffer::from_vec(values(19, 1));
        let mut gpu = MetalBuffer::from_vec(values(19, 1));
        let vector = values(19, 5);
        let cpu_vector = CpuBuffer::from_slice(&vector);
        let gpu_vector = MetalBuffer::from_slice(&vector);
        let weight = Field256::from(7);
        cpu_vector.mixed_scalar_mul_add_to(&Identity::new(), &mut cpu, weight);
        gpu_vector.mixed_scalar_mul_add_to(&Identity::new(), &mut gpu, weight);
        assert_eq!(gpu.as_slice(), cpu.as_slice());
    }

    #[test]
    fn metal_bn254_interleaved_rs_encode_matches_cpu() {
        // The GPU encoder and the CPU reference share one coset layout: the engine and the
        // domain are derived from the same field, and the GPU encode reads its layout from
        // `RsDomain::coset_params` (which the engine's slice encode also uses).
        let engine = NttEngine::<Field256>::new_from_fftfield();
        let gpu_rs = MetalRs::new(Arc::new(RsDomain::<Field256>::from_fftfield()));

        let a = values(8, 1);
        let gpu_a = MetalBuffer::from_slice(&a);
        let gpu_masks = MetalBuffer::from_slice(&[]);

        // CPU reference straight from the engine's slice API: `a` is one vector of two
        // length-4 messages, no masks, codeword length 8.
        let messages = a.chunks_exact(4).collect::<Vec<_>>();
        let cpu = engine.interleaved_encode_slices(&messages, &[], 8);
        let gpu = gpu_rs.interleaved_encode(&[&gpu_a], &gpu_masks, 4, 2, 8);
        assert_eq!(gpu.as_slice(), cpu.as_slice());
    }

    #[test]
    fn metal_bn254_mixed_extend_matches_cpu() {
        let values = values(8, 3);
        let point = vec![Field256::from(2), Field256::from(5), Field256::from(9)];
        let cpu = CpuBuffer::from_slice(&values);
        let gpu = MetalBuffer::from_slice(&values);
        assert_eq!(
            gpu.mixed_extend(&Identity::new(), &point),
            cpu.mixed_extend(&Identity::new(), &point)
        );
    }
}
