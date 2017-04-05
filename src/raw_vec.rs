
use ffi;
use super::util::FFTW_MUTEX;

use num_complex::Complex64 as c64;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::os::raw::c_void;
use std::ops::{Index, IndexMut};
use std::slice::{Iter, IterMut};

pub struct RawVec<T> {
    n: usize,
    data: *mut T,
}

impl<T> RawVec<T> {
    pub fn as_slice(&self) -> &[T] { unsafe { from_raw_parts(self.data, self.n) } }
    pub fn as_mut_slice(&mut self) -> &mut [T] { unsafe { from_raw_parts_mut(self.data, self.n) } }

    pub fn as_ptr(&self) -> *const T { self.data }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.data }

    pub fn len(&self) -> usize { self.n }

    pub fn iter<'a>(&'a self) -> Iter<'a, T> { self.as_slice().iter() }
    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, T> { self.as_mut_slice().iter_mut() }
}

impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        let lock = FFTW_MUTEX.lock().expect("Cannot get lock");
        unsafe { ffi::fftw_free(self.data as *mut c_void) };
        drop(lock);
    }
}

impl RawVec<f64> {
    pub fn new(n: usize) -> Self {
        let lock = FFTW_MUTEX.lock().expect("Cannot get lock");
        let ptr = unsafe { ffi::fftw_alloc_real(n) };
        drop(lock);
        RawVec { n: n, data: ptr }
    }
}

impl RawVec<c64> {
    pub fn new(n: usize) -> Self {
        let lock = FFTW_MUTEX.lock().expect("Cannot get lock");
        let ptr = unsafe { ffi::fftw_alloc_complex(n) } as *mut c64;
        drop(lock);
        RawVec { n: n, data: ptr }
    }
}

impl<T> Index<isize> for RawVec<T> {
    type Output = T;
    fn index(&self, index: isize) -> &Self::Output { unsafe { &*self.data.offset(index) } }
}

impl<T> IndexMut<isize> for RawVec<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output { unsafe { &mut *self.data.offset(index) } }
}
