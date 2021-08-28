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
pub mod upper_tri_dyn;
pub mod upper_tri_stc;

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

impl<T, D: SquareDimension> UpperTriRawData<T, D>
where
    T: Copy + Zero,
{
    fn data_size(&self) -> usize {
        let rank = self.rank.to_usize();
        rank * (rank + 1) / 2
    }

    fn get_offset(&self, row: usize, col: usize) -> Option<usize> {
        let rank = self.rank.to_usize();
        if row > col || col >= rank || row >= rank {
            return None;
        } else {
            return Some(offset_for_col(col, row));
        }
    }

    pub(crate) fn get_diag_el(&self, index: usize) -> Option<&T> {
        self.get(index, index)
    }

    pub(crate) fn get_diag_el_mut(&mut self, index: usize) -> Option<&mut T> {
        self.get_mut(index, index)
    }

    pub(crate) fn get(&self, row: usize, col: usize) -> Option<&T> {
        let offset = self.get_offset(row, col)?;
        self.buf.get(offset)
    }

    pub(crate) fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        let offset = self.get_offset(row, col)?;
        self.buf.get_mut(offset)
    }

    pub fn get_raw_col<'a>(&'a self, col: usize) -> ColView<&'a T, T, D> {
        ColView {
            ptr: self.buf.as_ptr(),
            col_num: col,
            row: 0,
            rank: self.rank,
            col_offset: col * (col + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub fn get_raw_col_mut<'a>(&'a mut self, col: usize) -> ColViewMut<&'a mut T, T, D> {
        ColViewMut {
            ptr: self.buf.as_mut_ptr(),
            col_num: col,
            row: 0,
            rank: self.rank,
            _ref_type: PhantomData,
            col_offset: col * (col + 1) / 2,
        }
    }

    pub fn get_raw_row<'a>(&'a self, row: usize) -> RowView<&'a T, T, D> {
        RowView {
            ptr: self.buf.as_ptr(),
            row_num: row,
            col: row + 1,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn get_raw_row_mut<'a>(&'a mut self, row: usize) -> RowViewMut<&'a mut T, T, D> {
        RowViewMut {
            ptr: self.buf.as_mut_ptr(),
            row_num: row,
            col: row + 1,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn get_corner<'a>(&'a self, diagonal_element: usize) -> CornerView<&'a T, T, D> {
        CornerView {
            ptr: self.buf.as_ptr(),
            diagonal_element,
            pos: 0,
            rank: self.rank,
            col_offset: diagonal_element * (diagonal_element + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub fn get_corner_mut<'a>(
        &'a mut self,
        diagonal_element: usize,
    ) -> CornerViewMut<&'a mut T, T, D> {
        CornerViewMut {
            ptr: self.buf.as_mut_ptr(),
            diagonal_element,
            pos: 0,
            rank: self.rank,
            col_offset: diagonal_element * (diagonal_element + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub fn get_diag(&self) -> DiagView<&T, T, D> {
        DiagView {
            ptr: self.buf.as_ptr(),
            pos: 0,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn get_diag_mut(&mut self) -> DiagViewMut<&mut T, T, D> {
        DiagViewMut {
            ptr: self.buf.as_mut_ptr(),
            pos: 0,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn map<B, F: FnMut(&T) -> B>(&self, f: F) -> UpperTriRawData<B, D> {
        let buf = self.buf.iter().map(f).collect();
        UpperTriRawData {
            buf,
            rank: self.rank,
        }
    }

    pub fn map_inplace<F: FnMut(&T) -> T>(&mut self, mut f: F) {
        self.buf.iter_mut().for_each(|t| {
            let new = f(t);
            *t = new;
        });
    }

    pub fn find_with_indices<Accum, Test, Accumulator>(
        &self,
        test: Test,
        accumulator: Accumulator,
    ) -> (IndexPair, Accum)
    where
        Accum: Default + Clone,
        Test: Fn(&Accum) -> bool,
        Accumulator: Fn(&T, &Accum) -> Accum,
    {
        let mut accum: Accum = Accum::default();
        let mut index_pair = IndexPair { row: 0, col: 0 };
        let _ = self
            .buf
            .iter()
            .map(|t| {
                let b = accumulator(t, &accum);
                accum = b.clone();
                b
            })
            .find(|b| {
                let test_val = test(b);
                if !test_val {
                    let IndexPair { row, col } = &mut index_pair;
                    if *row >= *col {
                        *row = 0;
                        *col += 1;
                    } else {
                        *row += 1
                    };
                }
                test_val
            });

        (index_pair, accum)
    }

    pub(crate) fn iter(&self) -> Iter<'_, T> {
        self.buf.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.buf.iter_mut()
    }

    pub(crate) fn into_iter(self) -> IntoIter<T> {
        self.buf.into_iter()
    }
}

impl<'a, T, D: SquareDimension> SubAssign<&'a UpperTriRawData<T, D>> for UpperTriRawData<T, D>
where
    T: for<'b> SubAssign<&'b T> + Copy + Zero,
{
    fn sub_assign(&mut self, rhs: &'a UpperTriRawData<T, D>) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(left, right)| *left -= right);
    }
}

impl<T, D: SquareDimension> SubAssign<UpperTriRawData<T, D>> for UpperTriRawData<T, D>
where
    T: SubAssign<T> + Copy + Zero,
{
    fn sub_assign(&mut self, rhs: UpperTriRawData<T, D>) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(left, right)| *left -= right);
    }
}

impl<T: Display + Zero + Copy, D: SquareDimension> Display for UpperTriRawData<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let space_len = format!("{} ", T::zero()).len();
        let space = repeat(" ").take(space_len).collect::<String>();
        write!(f, "\n")?;
        (0..self.rank.to_usize())
            .map(|row| -> Result<(), std::fmt::Error> {
                let spaces = repeat(space.clone()).take(row).collect::<String>();
                write!(f, "\t{}", spaces)?;
                let diag = self.get_diag_el(row).ok_or(std::fmt::Error)?;
                write!(f, "{} ", diag)?;
                self.get_raw_row(row)
                    .map(|t| -> Result<_, _> { write!(f, "{} ", t) })
                    .collect::<Result<Vec<()>, _>>()?;
                write!(f, "\n")
            })
            .collect::<Result<Vec<()>, std::fmt::Error>>()
            .map(|_| ())
    }
}

impl<'a, T, D: SquareDimension> AddAssign<&'a UpperTriRawData<T, D>> for UpperTriRawData<T, D>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    fn add_assign(&mut self, rhs: &'a UpperTriRawData<T, D>) {
        self.buf
            .iter_mut()
            .zip(rhs.buf.iter())
            .for_each(|(return_val, new)| {
                *return_val += new;
            });
    }
}

impl<T, D: SquareDimension> AddAssign<UpperTriRawData<T, D>> for UpperTriRawData<T, D>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    fn add_assign(&mut self, rhs: UpperTriRawData<T, D>) {
        *self += &rhs;
    }
}

impl<'a, T, D: SquareDimension> Add<&'a UpperTriRawData<T, D>> for UpperTriRawData<T, D>
where
    &'a UpperTriRawData<T, D>:
        for<'b> Add<&'b UpperTriRawData<T, D>, Output = UpperTriRawData<T, D>>,
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T, D>;

    fn add(self, rhs: &'a UpperTriRawData<T, D>) -> Self::Output {
        rhs + &self
    }
}

impl<'a, T, D: SquareDimension> Add<UpperTriRawData<T, D>> for &'a UpperTriRawData<T, D>
where
    &'a UpperTriRawData<T, D>:
        for<'b> Add<&'b UpperTriRawData<T, D>, Output = UpperTriRawData<T, D>>,
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T, D>;

    fn add(self, rhs: UpperTriRawData<T, D>) -> Self::Output {
        self + &rhs
    }
}

impl<T, D: SquareDimension> Add<UpperTriRawData<T, D>> for UpperTriRawData<T, D>
where
    UpperTriRawData<T, D>: for<'b> Add<&'b UpperTriRawData<T, D>, Output = UpperTriRawData<T, D>>,
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T, D>;

    fn add(self, rhs: UpperTriRawData<T, D>) -> Self::Output {
        self + &rhs
    }
}

#[cfg(test)]
mod test;
