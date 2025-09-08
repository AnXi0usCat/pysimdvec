#![feature(portable_simd)]
use std::simd::Simd;

type V = Simd<f32, 8>;

#[derive(Clone, Copy)]
enum Op {
    Add,
    Sub,
    Mul,
}

#[derive(Clone, Copy)]
enum Vs {
    Add,
    Mul,
    SubVecFromScalar,
    SubScalarFromVec,
}

#[inline]
fn scalar_bin(a: f32, b: f32, op: Op) -> f32 {
    match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
    }
}

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
                // a * 0.0 already zeroes out
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
            }
        }
    }

    out
}
