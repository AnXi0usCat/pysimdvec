from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="pysimdvec",
    version="0.0.1",
    packages=["pysimdvec"],
    rust_extensions=[
        RustExtension(
            "pysimdvec._pysimdvec",
            path="Cargo.toml",
            binding=Binding.Py03
        )
    ],
    zip_safe=False,
    description="Array type with SIMD-accelerated vector ops (+ padding) backed by Rust"
)
