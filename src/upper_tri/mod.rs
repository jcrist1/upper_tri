use crate::dimension::DynSquare;
use crate::dimension::SquareDimension;
use core::slice::Iter;
use rand_distr::num_traits::Zero;
use std::fmt::Display;
use std::iter::repeat;
use std::marker::PhantomData;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Deref;
use std::ops::Range;
use std::ops::SubAssign;
use std::slice::IterMut;
use std::vec::IntoIter;
pub mod upper_tri_dyn_alg;

#[derive(Clone)]
pub struct UpperTriRawData<T, D: SquareDimension + Clone> {
    buf: Vec<T>,
    pub rank: D,
}

// Need to guarantee that row <= col
#[derive(Clone, Copy)]
pub struct IndexPair {
    pub row: usize,
    pub col: usize,
}

fn offset_for_col(col: usize, row: usize) -> usize {
    let col_offset = (col * (col + 1)) / 2;
    col_offset + row
}

/// A struct for reading the component strictly above the diagonal
pub struct ColView<RefType, BaseType, D: SquareDimension> {
    ptr: *const BaseType,
    col_num: usize,
    rank: D,
    row: usize,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

/// A struct for writing the component strictly above the diagonal
pub struct ColViewMut<RefType, BaseType, D: SquareDimension> {
    ptr: *mut BaseType,
    col_num: usize,
    row: usize,
    rank: D,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a, D: SquareDimension> Iterator for ColViewMut<&'a mut T, T, D> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.col_num >= self.rank.to_usize() || self.row >= self.col_num {
            None
        } else {
            unsafe {
                let offset = self.col_offset + self.row;
                self.row += 1;
                Some(&mut *self.ptr.add(offset))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = 1 + self.col_num - self.row;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a, D: SquareDimension> Iterator for ColView<&'a T, T, D> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.col_num >= self.rank.to_usize() || self.row >= self.col_num {
            None
        } else {
            unsafe {
                let offset = self.col_offset + self.row;
                self.row += 1;
                Some(&*self.ptr.add(offset))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = 1 + self.col_num - self.row;
        (remaining, Some(remaining))
    }
}

/// A struct for reading the component strictly to the left of the diagonal
pub struct RowView<RefType, BaseType, D: SquareDimension> {
    ptr: *const BaseType,
    row_num: usize,
    rank: D,
    col: usize,
    _ref_type: PhantomData<RefType>,
}

/// A struct for writing the component strictly to the left of the diagonal
pub struct RowViewMut<RefType, BaseType, D: SquareDimension> {
    ptr: *mut BaseType,
    row_num: usize,
    col: usize,
    rank: D,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a, D: SquareDimension> Iterator for RowView<&'a T, T, D> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.col >= self.rank.to_usize() {
            None
        } else {
            unsafe {
                let offset = offset_for_col(self.col, self.row_num);
                self.col += 1;
                Some(&*self.ptr.add(offset))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rank.to_usize() - self.col;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a, D: SquareDimension> Iterator for RowViewMut<&'a mut T, T, D> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.col >= self.rank.to_usize() {
            None
        } else {
            unsafe {
                let offset = offset_for_col(self.col, self.row_num);
                self.col += 1;
                Some(&mut *self.ptr.add(offset))
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rank.to_usize() - self.col;
        (remaining, Some(remaining))
    }
}

pub struct CornerView<RefType, BaseType, D: SquareDimension> {
    ptr: *const BaseType,
    diagonal_element: usize,
    pos: usize,
    rank: D,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

pub struct CornerViewMut<RefType, BaseType, D: SquareDimension> {
    ptr: *mut BaseType,
    diagonal_element: usize,
    pos: usize,
    rank: D,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a, D: SquareDimension> Iterator for CornerViewMut<&'a mut T, T, D> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.diagonal_element >= self.rank.to_usize() {
            None
        } else if self.pos <= self.diagonal_element {
            unsafe {
                let offset = self.col_offset + self.pos;
                self.pos += 1;
                Some(&mut *self.ptr.add(offset))
            }
        } else if self.pos < self.rank.to_usize() {
            unsafe {
                let col = self.pos;
                let row = self.diagonal_element;
                let offset = offset_for_col(col, row);
                self.pos += 1;
                Some(&mut *self.ptr.add(offset))
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rank.to_usize() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a, D: SquareDimension> Iterator for CornerView<&'a T, T, D> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.diagonal_element >= self.rank.to_usize() {
            None
        } else if self.pos <= self.diagonal_element {
            unsafe {
                let offset = self.col_offset + self.pos;
                self.pos += 1;
                Some(&*self.ptr.add(offset))
            }
        } else if self.pos < self.rank.to_usize() {
            unsafe {
                let col = self.pos;
                let row = self.diagonal_element;
                let offset = offset_for_col(col, row);
                self.pos += 1;
                Some(&*self.ptr.add(offset))
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rank.to_usize() - self.pos;
        (remaining, Some(remaining))
    }
}

pub struct DiagView<RefType, BaseType, D: SquareDimension> {
    ptr: *const BaseType,
    pos: usize,
    rank: D,
    _ref_type: PhantomData<RefType>,
}

pub struct DiagViewMut<RefType, BaseType, D: SquareDimension> {
    ptr: *mut BaseType,
    pos: usize,
    rank: D,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a, D: SquareDimension> Iterator for DiagView<&'a T, T, D> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.pos >= self.rank.to_usize() {
            None
        } else if self.pos <= self.rank.to_usize() {
            unsafe {
                let offset = offset_for_col(self.pos, self.pos);
                self.pos += 1;
                Some(&*self.ptr.add(offset))
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rank.to_usize() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a, D: SquareDimension> Iterator for DiagViewMut<&'a mut T, T, D> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.pos >= self.rank.to_usize() {
            None
        } else if self.pos <= self.rank.to_usize() {
            unsafe {
                let offset = offset_for_col(self.pos, self.pos);
                self.pos += 1;
                Some(&mut *self.ptr.add(offset))
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.rank.to_usize() - self.pos;
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod test;
