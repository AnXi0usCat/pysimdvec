use pyo3::prelude::*;
use pyo3::types::PyList;

#[pyfunction]
fn add(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::add(&a, &b))
}

#[pyfunction]
fn sub(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::sub(&a, &b))
}

#[pyfunction]
fn mul(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::mul(&a, &b))
}

#[pyfunction]
fn add_scalar(v: Vec<f32>, s: f32) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::add_scalar(&v, s))
}

#[pyfunction]
fn sub_scalar(v: Vec<f32>, s: f32) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::sub_scalar(&v, s))
}

#[pyfunction]
fn scalar_sub(s: f32, v: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::scalar_sub(s, &v))
}

#[pyfunction]
fn mul_scalar(v: Vec<f32>, s: f32) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::mul_scalar(&v, s))
}

#[pymodule]
fn _pysimdvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(add_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(sub_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul_scalar, m)?)?;
    Ok(())
}

