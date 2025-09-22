# pysimdvec

## Build & try it

From the workspace root:

```bash
# optional: for best codegen on your Intel Mac
export RUSTFLAGS="-C target-cpu=native"

# run Rust tests
cargo test -p simdvec-core

# build and install the Python extension in editable mode
pip install maturin
maturin develop --release

# Python REPL
python - <<'PY'
import pysimdvec as sv
print(sv.add([1,2,3], [10,20]))
print(sv.sub([1,2,3], [10,20,30,40]))
print(sv.mul([1,2,3], [10,20]))
print(sv.mul_scalar([1,2,3], 2.5))
PY
