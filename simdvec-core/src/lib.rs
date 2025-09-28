#![feature(portable_simd)]
use core::f32;
use std::simd::Simd;

type V = Simd<f32, 8>;

/// Add two vectors (shorter is zero-padded).
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    unsafe {
    binop(a, b, Op::Add)
    }
}

/// Subtract two vectors (a - b; shorter is zero-padded).
pub fn sub(a: &[f32], b: &[f32]) -> Vec<f32> {
    unsafe {
    binop(a, b, Op::Sub)
    }
}

/// Multiply two vectors (shorter is zero-padded).
pub fn mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    unsafe {
        binop(a, b, Op::Mul)
    }
}

pub fn div(a: &[f32], b: &[f32]) -> Vec<f32> {
    unsafe {
        binop(a, b, Op::Div)
    }
}

/// Vector + scalar (element-wise).
pub fn add_scalar(v: &[f32], s: f32) -> Vec<f32> {
    unsafe {
    vec_scalar_op(v, s, Vs::Add)
    }
}

/// Vector - scalar (element-wise).
pub fn sub_scalar(v: &[f32], s: f32) -> Vec<f32> {
    unsafe {
    vec_scalar_op(v, s, Vs::SubScalarFromVec)
    }
}

/// Scalar - vector (element-wise): s - v[i].
pub fn scalar_sub(s: f32, v: &[f32]) -> Vec<f32> {
    unsafe {
    vec_scalar_op(v, s, Vs::SubVecFromScalar)
    }
}

/// Vector * scalar (element-wise).
pub fn mul_scalar(v: &[f32], s: f32) -> Vec<f32> {
    unsafe {
    vec_scalar_op(v, s, Vs::Mul)
    }
}

pub fn div_scalar(v: &[f32], s: f32) -> Vec<f32> {
    unsafe {
        vec_scalar_op(v, s, Vs::DivVecByScalar)
    }
}

pub fn scalar_div(s: f32, v: &[f32]) -> Vec<f32> {
    unsafe {
        vec_scalar_op(v, s, Vs::DivScalarByVec)
    }
}

#[derive(Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy)]
enum Vs {
    Add,
    Mul,
    SubVecFromScalar,
    SubScalarFromVec,
    DivVecByScalar,
    DivScalarByVec
}

#[inline]
fn scalar_bin(a: f32, b: f32, op: Op) -> f32 {
    match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
    }
}

#[target_feature(enable = "avx2")]
fn binop(a: &[f32], b: &[f32], op: Op) -> Vec<f32> {
    const LANES: usize = V::LEN;

    let a_len = a.len();
    let b_len = b.len();
    let n = a_len.max(b_len);
    let mut out = vec![0.0f32; n];

    // simd lenghts
    let common = a_len.min(b_len);
    let common_main = (common / LANES) * LANES;

    let (a_m, _) = a[..common_main].as_chunks::<LANES>();
    let (b_m, _) = b[..common_main].as_chunks::<LANES>();
    let (out_m, _) = out[..common_main].as_chunks_mut::<LANES>();

    // compute the min common mains
    for ((a8, b8), out8) in a_m.iter().zip(b_m).zip(out_m) {
        let va = V::from_array(*a8);
        let vb = V::from_array(*b8);

        let v_op = match op {
            Op::Add => va + vb,
            Op::Sub => va - vb,
            Op::Mul => va * vb,
            Op::Div => va / vb,
        };
        *out8 = v_op.to_array();
    }

    // tail over common region
    for i in common_main..common {
        out[i] = scalar_bin(a[i], b[i], op);
    }

    // remainder where one side is shorter then the rest
    if a_len > b_len {
        match op {
            Op::Add | Op::Sub => {
                out[common..].copy_from_slice(&a[common..]);
            },
            Op::Mul => {
                // a * 0.0 already zeroed out
            },
            Op::Div => {
                out[common..].copy_from_slice(&vec![f32::NAN; a.len() - common]);
            }
        }
    } else if b_len > a_len {
        match op {
            Op::Add => {
                // 0.0 + b = b
                out[common..].copy_from_slice(&b[common..]);
            },
            Op::Sub => {
                // 0.0 - b = -b
                for (o, &bv) in out[common..].iter_mut().zip(&b[common..]) {
                    *o = -bv;
                }
            },
            Op::Mul => {
                // b * 0.0 already zeroed out
            },
            Op::Div => {
                // 0.0 / b already zeroed out
            }
        }
    }
    out
}

#[target_feature(enable = "avx2")]
fn vec_scalar_op(v: &[f32], sc: f32, kind: Vs) -> Vec<f32> {
    const LANES: usize = V::LEN;

    let n = v.len();
    let mut out = vec![0.0f32; n];
    let main = (n / LANES) * LANES;
    let vs = V::splat(sc);

    let (v_main, _) = v[..main].as_chunks::<LANES>();
    let (out_main, _) = out[..main].as_chunks_mut::<LANES>();

    // main logic
    for (v8, o8) in v_main.iter().zip(out_main) {
        let vv = V::from_array(*v8);

        let v_op = match kind {
            Vs::Add => vv + vs,
            Vs::SubVecFromScalar => vs - vv,
            Vs::SubScalarFromVec => vv - vs,
            Vs::Mul => vv * vs,
            Vs::DivVecByScalar => vv / vs,
            Vs::DivScalarByVec => vs / vv,
        };

        *o8 = v_op.to_array();
    }

    // remainder logic
    for i in main..n {
        out[i] = match kind {
            Vs::Add => v[i] + sc,
            Vs::Mul => v[i] * sc,
            Vs::SubScalarFromVec => v[i] - sc,
            Vs::SubVecFromScalar => sc - v[i],
            Vs::DivVecByScalar => v[i] / sc,
            Vs::DivScalarByVec => sc / v[i],
        };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binop_padding() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0];
        assert_eq!(add(&a, &b), vec![11.0, 22.0, 3.0, 4.0]);  // add, leftover a
        assert_eq!(sub(&a, &b), vec![ -9.0, -18.0, 3.0, 4.0]); // a-b, leftover a
        assert_eq!(mul(&a, &b), vec![10.0, 40.0, 0.0, 0.0]);   // a*b, leftover zeros
        // test division
        for (&test, expected) in div(&a, &b).iter().zip(vec![0.1, 0.1, f32::NAN, f32::NAN]) {
            if test.is_nan() && expected.is_nan() {} // NAN is not equal to anything
            else {
                assert_eq!(test, expected);
            }
        } 
        let c = vec![5.0];
        assert_eq!(sub(&c, &a), vec![4.0, -2.0, -3.0, -4.0]); // 0 - a tail
        assert_eq!(div(&c, &a), vec![5.0, 0.0, 0.0, 0.0]); // 0 - a tail
    }

    #[test]
    fn vec_scalar() {
        let v = vec![1.0, 2.0, 3.0];
        assert_eq!(add_scalar(&v, 2.0), vec![3.0, 4.0, 5.0]);
        assert_eq!(sub_scalar(&v, 2.0), vec![-1.0, 0.0, 1.0]);
        assert_eq!(scalar_sub(2.0, &v), vec![1.0, 0.0, -1.0]);
        assert_eq!(mul_scalar(&v, 3.0), vec![3.0, 6.0, 9.0]);
        assert_eq!(div_scalar(&v, 1.0), vec![1.0, 2.0, 3.0]);
        assert_eq!(scalar_div(1.0, &v), vec![1.0, 0.5, 0.33333334]);
    }
}
