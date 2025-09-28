use std::usize;

use pyo3::exceptions::{PyIndexError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyType};

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
fn div(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::div(&a, &b))
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

#[pyfunction]
fn div_scalar(v: Vec<f32>, s: f32) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::div_scalar(&v, s))
}

#[pyfunction]
fn scalar_div(s: f32, v: Vec<f32>) -> PyResult<Vec<f32>> {
    Ok(simdvec_core::scalar_div(s, &v))
}

#[pyclass(module = "pysimdvec._pysimdvec")]
#[derive(Clone)]
pub struct Array {
    data: Vec<f32>
}

#[pymethods]
impl Array {

    #[new]
    fn new(obj: &Bound<PyAny>) -> PyResult<Self> {
        let v: Vec<f32> = obj.extract::<Vec<f32>>()?;
        Ok(Self { data: v })
    }

    #[classmethod]
    fn zeroes(_cls: &Bound<'_, PyType>, n: usize) -> PyResult<Self> {
        Ok(Self { data: vec![0.0; n] })
    }

    fn tolist(&self) -> Vec<f32> {
        self.data.clone()
    }

    fn __len__(&self) -> usize {
        self.data.len()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Array({:?})", self.data))
    }

    fn __getitem__(&self, idx: isize) -> PyResult<f32> {
        if idx < 0 || idx >= self.data.len() as isize {
            return Err(PyIndexError::new_err("Index out of range"));
        }
        Ok(self.data[idx as usize])
    }

    fn __add__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(o) = rhs.extract::<Array>() {
            Ok(Self { data: simdvec_core::add(&self.data, &o.data) })
        } else if let Ok(s) = rhs.extract::<f32>() {
            Ok(Self { data: simdvec_core::add_scalar(&self.data, s)})
        } else if let Ok(s) = rhs.extract::<i64>() {
            Ok(Self { data: simdvec_core::add_scalar(&self.data, s as f32)})
        } else {
            Err(PyTypeError::new_err("__add__ expects Array or number"))
        }
    }

    fn __radd__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(lhs)
    }

    fn __sub__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(o) = rhs.extract::<Array>() {
            Ok(Self { data: simdvec_core::sub(&self.data, &o.data) })
        } else if let Ok(s) = rhs.extract::<f32>() {
            Ok(Self { data: simdvec_core::sub_scalar(&self.data, s)})
        } else if let Ok(s) = rhs.extract::<i64>() {
            Ok(Self { data: simdvec_core::sub_scalar(&self.data, s as f32)})
        } else {
            Err(PyTypeError::new_err("__sub__ expects Array or number"))
        }
    }

    fn __rsub__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(o) = lhs.extract::<Array>() {
            Ok(Self { data: simdvec_core::sub(&o.data, &self.data) })
        } else if let Ok(s) = lhs.extract::<f32>() {
            Ok(Self { data: simdvec_core::scalar_sub(s, &self.data)})
        } else if let Ok(s) = lhs.extract::<i64>() {
            Ok(Self { data: simdvec_core::scalar_sub(s as f32, &self.data)})
        } else {
            Err(PyTypeError::new_err("__rsub__ expects Array or number"))
        }
    }

    fn __mul__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(o) = rhs.extract::<Array>() {
            Ok(Self { data: simdvec_core::mul(&self.data, &o.data) })
        } else if let Ok(s) = rhs.extract::<f32>() {
            Ok(Self { data: simdvec_core::mul_scalar(&self.data, s)})
        } else if let Ok(s) = rhs.extract::<i64>() {
            Ok(Self { data: simdvec_core::mul_scalar(&self.data, s as f32)})
        } else {
            Err(PyTypeError::new_err("__mul__ expects Array or number"))
        }
    }

    fn __rmul__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(lhs)
    }

    fn __truediv__(&self, rhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(o) = rhs.extract::<Array>() {
            Ok(Self { data: simdvec_core::div(&self.data, &o.data)})
        } else if let Ok(s) = rhs.extract::<f32>() {
            Ok(Self { data: simdvec_core::div_scalar(&self.data, s)})
        } else if let Ok(s) = rhs.extract::<i64>() {
            Ok(Self { data: simdvec_core::div_scalar(&self.data, s as f32)})
        } else {
            Err(PyTypeError::new_err("__div__ expects array or number"))
        }
    }
    
    fn __rtruediv__(&self, lhs: &Bound<PyAny>) -> PyResult<Self> {
        if let Ok(o) = lhs.extract::<Array>() {
            Ok(Self { data: simdvec_core::div(&o.data, &self.data)})
        } else if let Ok(s) = lhs.extract::<f32>() {
            Ok(Self { data: simdvec_core::scalar_div(s, &self.data)})
        } else if let Ok(s) = lhs.extract::<i64>() {
            Ok(Self { data: simdvec_core::scalar_div(s as f32, &self.data)})
        } else {
            Err(PyTypeError::new_err("__rdiv__ expects array or number"))
        }
    }
}

#[pymodule]
fn _pysimdvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul, m)?)?;
    m.add_function(wrap_pyfunction!(div, m)?)?;
    m.add_function(wrap_pyfunction!(add_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(sub_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_sub, m)?)?;
    m.add_function(wrap_pyfunction!(mul_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(div_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(scalar_div, m)?)?;
    m.add_class::<Array>()?;
    Ok(())
}

