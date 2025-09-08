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
