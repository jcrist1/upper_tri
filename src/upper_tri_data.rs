use core::slice::Iter;
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

use rand_distr::num_traits::Zero;

#[derive(Clone)]
pub struct UpperTriRawData<T> {
    buf: Vec<T>,
    pub rank: usize,
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
pub struct ColView<RefType, BaseType> {
    ptr: *const BaseType,
    col_num: usize,
    rank: usize,
    row: usize,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

/// A struct for writing the component strictly above the diagonal
pub struct ColViewMut<RefType, BaseType> {
    ptr: *mut BaseType,
    col_num: usize,
    row: usize,
    rank: usize,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a> Iterator for ColViewMut<&'a mut T, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.col_num >= self.rank || self.row >= self.col_num {
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

impl<'a, T: 'a> Iterator for ColView<&'a T, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.col_num >= self.rank || self.row >= self.col_num {
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
pub struct RowView<RefType, BaseType> {
    ptr: *const BaseType,
    row_num: usize,
    rank: usize,
    col: usize,
    _ref_type: PhantomData<RefType>,
}

/// A struct for writing the component strictly to the left of the diagonal
pub struct RowViewMut<RefType, BaseType> {
    ptr: *mut BaseType,
    row_num: usize,
    col: usize,
    rank: usize,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a> Iterator for RowView<&'a T, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.col >= self.rank {
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
        let remaining = self.rank - self.col;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a> Iterator for RowViewMut<&'a mut T, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.col >= self.rank {
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
        let remaining = self.rank - self.col;
        (remaining, Some(remaining))
    }
}

pub struct CornerView<RefType, BaseType> {
    ptr: *const BaseType,
    diagonal_element: usize,
    pos: usize,
    rank: usize,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

pub struct CornerViewMut<RefType, BaseType> {
    ptr: *mut BaseType,
    diagonal_element: usize,
    pos: usize,
    rank: usize,
    col_offset: usize,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a> Iterator for CornerViewMut<&'a mut T, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.diagonal_element >= self.rank {
            None
        } else if self.pos <= self.diagonal_element {
            unsafe {
                let offset = self.col_offset + self.pos;
                self.pos += 1;
                Some(&mut *self.ptr.add(offset))
            }
        } else if self.pos < self.rank {
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
        let remaining = self.rank - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a> Iterator for CornerView<&'a T, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.diagonal_element >= self.rank {
            None
        } else if self.pos <= self.diagonal_element {
            unsafe {
                let offset = self.col_offset + self.pos;
                self.pos += 1;
                Some(&*self.ptr.add(offset))
            }
        } else if self.pos < self.rank {
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
        let remaining = self.rank - self.pos;
        (remaining, Some(remaining))
    }
}

pub struct DiagView<RefType, BaseType> {
    ptr: *const BaseType,
    pos: usize,
    rank: usize,
    _ref_type: PhantomData<RefType>,
}

pub struct DiagViewMut<RefType, BaseType> {
    ptr: *mut BaseType,
    pos: usize,
    rank: usize,
    _ref_type: PhantomData<RefType>,
}

impl<'a, T: 'a> Iterator for DiagView<&'a T, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.pos >= self.rank {
            None
        } else if self.pos <= self.rank {
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
        let remaining = self.rank - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a, T: 'a> Iterator for DiagViewMut<&'a mut T, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<&'a mut T> {
        if self.pos >= self.rank {
            None
        } else if self.pos <= self.rank {
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
        let remaining = self.rank - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T> UpperTriRawData<T>
where
    T: Copy + Zero,
{
    fn data_size(&self) -> usize {
        self.rank * (self.rank + 1) / 2
    }
    pub fn new(rank: usize) -> Self {
        let def = T::zero();
        let buf = repeat(def).take(rank * (rank + 1) / 2).collect();
        Self { buf, rank }
    }

    fn get_offset(&self, row: usize, col: usize) -> Option<usize> {
        if row > col || col >= self.rank || row >= self.rank {
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

    fn get(&self, row: usize, col: usize) -> Option<&T> {
        let offset = self.get_offset(row, col)?;
        self.buf.get(offset)
    }

    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        let offset = self.get_offset(row, col)?;
        self.buf.get_mut(offset)
    }

    pub(crate) fn get_raw_col<'a>(&'a self, col: usize) -> ColView<&'a T, T> {
        ColView {
            ptr: self.buf.as_ptr(),
            col_num: col,
            row: 0,
            rank: self.rank,
            col_offset: col * (col + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub(crate) fn get_raw_col_mut<'a>(&'a mut self, col: usize) -> ColViewMut<&'a mut T, T> {
        ColViewMut {
            ptr: self.buf.as_mut_ptr(),
            col_num: col,
            row: 0,
            rank: self.rank,
            _ref_type: PhantomData,
            col_offset: col * (col + 1) / 2,
        }
    }

    pub(crate) fn get_raw_row<'a>(&'a self, row: usize) -> RowView<&'a T, T> {
        RowView {
            ptr: self.buf.as_ptr(),
            row_num: row,
            col: row + 1,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub(crate) fn get_raw_row_mut<'a>(&'a mut self, row: usize) -> RowViewMut<&'a mut T, T> {
        RowViewMut {
            ptr: self.buf.as_mut_ptr(),
            row_num: row,
            col: row + 1,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub(crate) fn get_corner<'a>(&'a self, diagonal_element: usize) -> CornerView<&'a T, T> {
        CornerView {
            ptr: self.buf.as_ptr(),
            diagonal_element,
            pos: 0,
            rank: self.rank,
            col_offset: diagonal_element * (diagonal_element + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub(crate) fn get_corner_mut<'a>(
        &'a mut self,
        diagonal_element: usize,
    ) -> CornerViewMut<&'a mut T, T> {
        CornerViewMut {
            ptr: self.buf.as_mut_ptr(),
            diagonal_element,
            pos: 0,
            rank: self.rank,
            col_offset: diagonal_element * (diagonal_element + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub(crate) fn get_diag(&self) -> DiagView<&T, T> {
        DiagView {
            ptr: self.buf.as_ptr(),
            pos: 0,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub(crate) fn get_diag_mut(&mut self) -> DiagViewMut<&mut T, T> {
        DiagViewMut {
            ptr: self.buf.as_mut_ptr(),
            pos: 0,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub(crate) fn push_final_col_iter<DerefT: Deref<Target = T>, Itr: Iterator<Item = DerefT>>(
        &mut self,
        iter: Itr,
    ) {
        let new = self.rank + 1;

        let default = T::zero();
        let new_iter = iter.map(|x| *x).chain(repeat(default)).take(new);

        self.buf.reserve(new); //Could probably bypass this if we can assure that new iter has trusted_len
        self.buf.extend(new_iter);
        self.rank += 1;
    }

    pub(crate) fn push_final_col_iter_owned<Itr: Iterator<Item = T>>(&mut self, iter: Itr) {
        let new = self.rank + 1;

        let default = T::zero();
        let new_iter = iter.map(|x| x).chain(repeat(default)).take(new);

        self.buf.reserve(new); //Could probably bypass this if we can assure that new iter has trusted_len
        self.buf.extend(new_iter);
        self.rank += 1;
    }

    pub fn push_final_col(&mut self, vec: &[T]) {
        self.push_final_col_iter(vec.iter())
    }

    pub fn map<B, F: FnMut(&T) -> B>(&self, f: F) -> UpperTriRawData<B> {
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

    pub(crate) fn drop_at(&mut self, index: usize) -> Vec<T> {
        let mut return_vec = Vec::with_capacity(self.rank);
        let col_offset = index * (index + 1) / 2;
        let col_end = col_offset + index + 1;
        let tmp = self.buf.drain(Range {
            start: col_offset,
            end: col_end,
        });

        return_vec.extend(tmp);
        let mut removed = index;
        for col in (index + 1)..self.rank {
            let next_to_remove = offset_for_col(col, index) - removed;
            let t = self.buf.remove(next_to_remove);
            return_vec.push(t);
            removed += 1;
        }
        self.rank -= 1;
        return_vec
    }

    pub(crate) fn find_with_indices<Accum, Test, Accumulator>(
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

impl<'a, T> Add<&'a UpperTriRawData<T>> for &'a UpperTriRawData<T>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T>;

    fn add(self, rhs: &'a UpperTriRawData<T>) -> Self::Output {
        if rhs.rank >= self.rank {
            let mut return_buf = rhs.buf.clone();
            return_buf
                .iter_mut()
                .zip(self.buf.iter())
                .for_each(|(return_val, new)| {
                    *return_val += new;
                });

            UpperTriRawData {
                buf: return_buf,
                rank: rhs.rank,
            }
        } else {
            let mut return_buf = self.buf.clone();
            return_buf
                .iter_mut()
                .zip(rhs.buf.iter())
                .for_each(|(return_val, new)| {
                    *return_val += new;
                });
            UpperTriRawData {
                buf: return_buf,
                rank: self.rank,
            }
        }
    }
}

impl<'a, T> SubAssign<&'a UpperTriRawData<T>> for UpperTriRawData<T>
where
    T: for<'b> SubAssign<&'b T> + Copy + Zero,
{
    fn sub_assign(&mut self, rhs: &'a UpperTriRawData<T>) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(left, right)| *left -= right);
    }
}

impl<T> SubAssign<UpperTriRawData<T>> for UpperTriRawData<T>
where
    T: SubAssign<T> + Copy + Zero,
{
    fn sub_assign(&mut self, rhs: UpperTriRawData<T>) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(left, right)| *left -= right);
    }
}

impl<T: Display + Zero + Copy> Display for UpperTriRawData<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let space_len = format!("{} ", T::zero()).len();
        let space = repeat(" ").take(space_len).collect::<String>();
        write!(f, "\n")?;
        (0..self.rank)
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

impl<'a, T> Add<&'a UpperTriRawData<T>> for UpperTriRawData<T>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T>;

    fn add(self, rhs: &'a UpperTriRawData<T>) -> Self::Output {
        &self + rhs
    }
}

impl<'a, T> Add<UpperTriRawData<T>> for &'a UpperTriRawData<T>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T>;

    fn add(self, rhs: UpperTriRawData<T>) -> Self::Output {
        self + &rhs
    }
}

impl<'a, T> Add<UpperTriRawData<T>> for UpperTriRawData<T>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T>;

    fn add(self, rhs: UpperTriRawData<T>) -> Self::Output {
        &self + &rhs
    }
}

impl<'a, T> AddAssign<&'a UpperTriRawData<T>> for UpperTriRawData<T>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    fn add_assign(&mut self, rhs: &'a UpperTriRawData<T>) {
        self.buf
            .iter_mut()
            .zip(rhs.buf.iter())
            .for_each(|(return_val, new)| {
                *return_val += new;
            });
    }
}

impl<T> AddAssign<UpperTriRawData<T>> for UpperTriRawData<T>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    fn add_assign(&mut self, rhs: UpperTriRawData<T>) {
        *self += &rhs;
    }
}

#[cfg(test)]
mod test {
    use crate::upper_tri_data::UpperTriRawData;
    use std::iter::repeat;

    #[test]
    fn test_some_junk() {
        let hundurd: usize = 100;
        let mut b = UpperTriRawData::<f32>::new(hundurd);

        let mut val = 0.0;
        let row_index = 53;
        b.get_raw_row_mut(row_index).for_each(|x| {
            *x = val;
            val += 1.0;
        });
        let c = b.get_raw_row(row_index).map(|x| *x).collect::<Vec<_>>();
        assert_eq!(
            c,
            (0..(hundurd - row_index))
                .map(|x| x as f32)
                .collect::<Vec<_>>()
        );
        let d = b.get_raw_row(hundurd - 1).collect::<Vec<_>>();
        assert_eq!(d.len(), 1);
        let e = b.get_raw_col(0).collect::<Vec<_>>();
        assert_eq!(e.len(), 1);
        let row = b.get_raw_row(0).collect::<Vec<_>>();
        assert_eq!(row.len(), hundurd);
        let col = b.get_raw_col(hundurd - 1).collect::<Vec<_>>();
        assert_eq!(col.len(), hundurd);
    }

    #[test]
    fn test_element_access() {
        let mut upper_tri = UpperTriRawData::<usize>::new(0);
        for i in 0..10 {
            let iter = (0..(i + 1)).rev();
            upper_tri.push_final_col_iter_owned(iter);
        }

        for col in 0..10 {
            for row in 0..(col + 1) {
                assert_eq!(upper_tri.get(row, col), Some(&(col - row)));
            }
        }
    }

    #[test]
    /// We're testing that if we access out of bounds we get empty iterators
    /// we also test that if we're in bounds we've got the right limits
    fn test_row_col_access() {
        let mut upper_tri = UpperTriRawData::<usize>::new(10);
        let x: i32 = upper_tri.get_raw_col(1).map(|_| 1).sum();
        assert_eq!(x, 2); // index 1 is the second col ... arrays start at 0

        let x: i32 = upper_tri.get_raw_col(11).map(|_| 1).sum();
        assert_eq!(x, 0);

        let x: i32 = upper_tri.get_raw_col_mut(1).map(|_| 1).sum();
        assert_eq!(x, 2);

        let x: i32 = upper_tri.get_raw_col_mut(11).map(|_| 1).sum();
        assert_eq!(x, 0);

        let x: i32 = upper_tri.get_raw_row(11).map(|_| 1).sum();
        assert_eq!(x, 0);

        let x: i32 = upper_tri.get_raw_row(0).map(|_| 1).sum();
        assert_eq!(x, 10);
        let x: i32 = upper_tri.get_raw_row_mut(11).map(|_| 1).sum();
        assert_eq!(x, 0);

        let x: i32 = upper_tri.get_raw_row_mut(0).map(|_| 1).sum();
        assert_eq!(x, 10);

        let x: i32 = upper_tri.get_corner(0).map(|_| 1).sum();
        assert_eq!(x, 10);

        let x: i32 = upper_tri.get_corner_mut(4).map(|_| 1).sum();
        assert_eq!(x, 10);

        let x: i32 = upper_tri.get_corner_mut(12).map(|_| 1).sum();
        assert_eq!(x, 0);
    }

    #[test]
    fn test_creation() {
        let ten: usize = 10;
        let mut upper_tri = UpperTriRawData::<usize>::new(ten);
        assert_eq!(upper_tri.buf.len(), 10 * 11 / 2);
        let vec = (0..11).collect::<Vec<_>>();
        upper_tri.push_final_col(&vec);
        // dropping the "corner" at the fifth element
        // (arrays start at 0)
        upper_tri.drop_at(4);
        let col = upper_tri.get_raw_col(9).map(|x| *x).collect::<Vec<_>>();
        assert_eq!(vec![0, 1, 2, 3, 4, 6, 7, 8, 9, 10], col);
    }
    #[test]
    fn test_addition() {
        let ten: usize = 10;
        let mut upper_tri_1 = UpperTriRawData::<isize>::new(1);
        let mut upper_tri_2 = UpperTriRawData::<isize>::new(1);
        upper_tri_1.push_final_col_iter(repeat(&(-1)));
        upper_tri_1.push_final_col_iter(repeat(&1));
        upper_tri_2.push_final_col_iter(repeat(&1));
        upper_tri_2.push_final_col_iter(repeat(&1));

        let upper_tri_3 = upper_tri_1 + upper_tri_2;
        let second_col = upper_tri_3.get_raw_col(1).map(|x| *x).collect::<Vec<_>>();
        let third_col = upper_tri_3.get_raw_col(2).map(|x| *x).collect::<Vec<_>>();
        assert_eq!(second_col, vec![0, 0]);
        assert_eq!(third_col, vec![2, 2, 2]);
    }
}
