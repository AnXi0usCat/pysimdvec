#![feature(portable_simd)]
use std::simd::Simd;

type V = Simd<f32, 8>;

#[derive(Clone,Copy)]
enum Op {
    Add,
    Sub,
    Mul
}

#[derive(Clone, Copy)]
enum Vs {
    Add,
    Mul,
    SubVecFromScalar,
    SubScalarFromVec,
}

#[inline]
fn scalar_bin(a: f32, b:f32, op: Op) -> f32 {
    match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
    }
}
