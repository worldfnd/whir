// NOTE: 100% AI GENERATED

pub(crate) const METAL_SOURCE: &str = r#"
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

kernel void bn254_scalar_mul(
    device ulong *acc [[buffer(0)]],
    device const ulong *weight_buf [[buffer(1)]],
    constant uint &len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) return;
    F weight = load_f(weight_buf, 0);
    store_f(acc, gid, mul_f(weight, load_f(acc, gid)));
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

pub(crate) const REDUCTION_CHUNK_SIZE: usize = 64;
pub(crate) const SMALL_GEOMETRIC_ACCUMULATE_CHUNK_SIZE: usize = 8;
pub(crate) const LARGE_GEOMETRIC_ACCUMULATE_CHUNK_SIZE: usize = 32;
/// Must match the `sums` register array size in the strided kernel.
pub(crate) const MAX_GEOMETRIC_ACCUMULATE_CHUNK_SIZE: usize = 32;
pub(crate) const LARGE_GEOMETRIC_ACCUMULATE_THRESHOLD: usize = 1 << 18;
pub(crate) const GEOMETRIC_ACCUMULATE_POINT_BLOCK_SIZE: usize = 16;
pub(crate) const GEOMETRIC_ACCUMULATE_POINT_BLOCK_THRESHOLD: usize = 1 << 17;
pub(crate) const GEOMETRIC_ACCUMULATE_POINT_BLOCK_MIN_POINTS: usize = 64;
pub(crate) const GEOMETRIC_ACCUMULATE_POINT_BLOCK_BATCH_BYTES: usize = 256 << 20;
