# pysimdvec

SIMD-accelerated vector math for Python, backed by Rust.
Provides a lightweight Array type that supports +, -, * with other Arrays or scalars.
When arrays have different lengths, the shorter one is padded with zeros.

Fast inner loops implemented in Rust using std::simd

Clean Python API with operator overloading

No maturin required — built like a “normal” Python package via setuptools-rust

## Features

Array([..]) – hold a 1D float array (f32 under the hood)

Binary ops:

a + b, a - b, a * b where a, b are Arrays

Mismatched lengths → shorter side is treated as zeros

Scalar ops (both orders):

a + 5, 5 + a

a - 5, 5 - a

a * 3, 3 * a

Helpers:

Array.zeros(n) – construct zeros

arr.tolist() – return a Python list

len(arr), indexing (arr[i])

Padding semantics:

add: leftover = the longer side

sub: a - b; leftover on a stays as a, leftover on b becomes -b

mul: leftover = 0

## Project layout
pysimdvec/
├─ setup.py
├─ pyproject.toml
├─ Cargo.toml                 # Rust extension crate (PyO3)
├─ src/lib.rs                 # PyO3 module: Array type + operators
├─ simdvec_core/              # pure Rust core (SIMD helpers)
│  ├─ Cargo.toml
│  └─ src/lib.rs
└─ pysimdvec/
   └─ __init__.py             # Python package that exposes Array


simdvec_core contains portable SIMD code (std::simd), tested in Rust.

The top-level Rust crate builds the Python extension module pysimdvec._pysimdvec.

## Requirements

Python 3.11+

Rust toolchain (stable): curl https://sh.rustup.rs -sSf | sh

Build deps: pip install -U pip setuptools wheel setuptools-rust

On macOS, ensure Xcode command line tools are installed:

xcode-select --install

## Install (editable dev mode)

From the project root:
```
pip install -U pip setuptools wheel setuptools-rust

# (optional) better codegen on your CPU
export RUSTFLAGS="-C target-cpu=native"

pip install -e .
```

This compiles the Rust extension and installs the pysimdvec Python package.

## Quickstart
```
from pysimdvec import Array

a = Array([1, 2, 3])
b = Array([10, 20])

print((a + b).tolist())   # [11.0, 22.0, 3.0]   (b is padded with 0 at the end)
print((a - b).tolist())   # [-9.0, -18.0, 3.0]
print((b - a).tolist())   # [9.0, 18.0, -3.0]
print((a * b).tolist())   # [10.0, 40.0, 0.0]

# Scalars on either side
print((a + 5).tolist())   # [6.0, 7.0, 8.0]
print((5 + a).tolist())   # [6.0, 7.0, 8.0]
print((a - 10).tolist())  # [-9.0, -8.0, -7.0]
print((10 - a).tolist())  # [9.0, 8.0, 7.0]
print((a * 3).tolist())   # [3.0, 6.0, 9.0]
print((3 * a).tolist())   # [3.0, 6.0, 9.0]

# Helpers
z = Array.zeros(4)
print(len(z), z.tolist()) # 4 [0.0, 0.0, 0.0, 0.0]
print(a[0], a[-1])        # 1.0 3.0
```
## Performance tips

Build with your native CPU features (AVX2/FMA where available):
```
export RUSTFLAGS="-C target-cpu=native"
pip install -e .
```

The core uses std::simd with f32 lanes (8-wide on AVX2).

Expect larger speedups on compute-dense kernels; pure element-wise ops can be memory-bandwidth bound.

Developing
Run Rust unit tests (core)
```
cargo test -p simdvec_core
```

Reinstall the extension after changes
```
# from project root
pip install -e .
```
Build a wheel (optional)
```
pip wheel .
```
## API reference (Python)
class Array(seq | iterable)

Create from any Python sequence of numbers (converted to float).

Array.zeros(n: int) -> Array

tolist() -> list[float]

__len__() -> int

__getitem__(i: int) -> float (supports negative indices)

Operators

Array + Array → zero-padded add

Array - Array → zero-padded subtract (a - b)

Array * Array → zero-padded multiply

Array + (int|float) and (int|float) + Array

Array - (int|float) and (int|float) - Array

Array * (int|float) and (int|float) * Array

In-place variants (+=, -=, *=) aren’t implemented yet.

## Troubleshooting

error: can't find Rust compiler
Install Rust: rustup (see Requirements).

macOS linker issues
Make sure command line tools are present: xcode-select --install.

Slow builds / unexpected codegen
Use export RUSTFLAGS="-C target-cpu=native" before pip install -e ..

## Roadmap

f64 support (likely Simd<f64, 4>)

In-place operators (__iadd__, __isub__, __imul__)

NumPy zero-copy interop (pyo3-numpy)

Matrix primitives (for the larger SIMD learning project)

## License

Dual-licensed under MIT or Apache-2.0.
