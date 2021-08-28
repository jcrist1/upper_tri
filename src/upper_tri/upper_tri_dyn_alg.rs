use crate::upper_tri::*;

impl<T> UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero,
{
    fn data_size(&self) -> usize {
        let DynSquare(rank) = self.rank;
        rank * (rank + 1) / 2
    }
    pub fn new(rank: usize) -> Self {
        let def = T::zero();
        let buf = repeat(def).take(rank * (rank + 1) / 2).collect();
        Self {
            buf,
            rank: DynSquare(rank),
        }
    }

    fn get_offset(&self, row: usize, col: usize) -> Option<usize> {
        let DynSquare(rank) = self.rank;
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

    pub fn get_raw_col<'a>(&'a self, col: usize) -> ColView<&'a T, T, DynSquare> {
        ColView {
            ptr: self.buf.as_ptr(),
            col_num: col,
            row: 0,
            rank: self.rank,
            col_offset: col * (col + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub fn get_raw_col_mut<'a>(&'a mut self, col: usize) -> ColViewMut<&'a mut T, T, DynSquare> {
        ColViewMut {
            ptr: self.buf.as_mut_ptr(),
            col_num: col,
            row: 0,
            rank: self.rank,
            _ref_type: PhantomData,
            col_offset: col * (col + 1) / 2,
        }
    }

    pub fn get_raw_row<'a>(&'a self, row: usize) -> RowView<&'a T, T, DynSquare> {
        RowView {
            ptr: self.buf.as_ptr(),
            row_num: row,
            col: row + 1,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn get_raw_row_mut<'a>(&'a mut self, row: usize) -> RowViewMut<&'a mut T, T, DynSquare> {
        RowViewMut {
            ptr: self.buf.as_mut_ptr(),
            row_num: row,
            col: row + 1,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn get_corner<'a>(&'a self, diagonal_element: usize) -> CornerView<&'a T, T, DynSquare> {
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
    ) -> CornerViewMut<&'a mut T, T, DynSquare> {
        CornerViewMut {
            ptr: self.buf.as_mut_ptr(),
            diagonal_element,
            pos: 0,
            rank: self.rank,
            col_offset: diagonal_element * (diagonal_element + 1) / 2,
            _ref_type: PhantomData,
        }
    }

    pub fn get_diag(&self) -> DiagView<&T, T, DynSquare> {
        DiagView {
            ptr: self.buf.as_ptr(),
            pos: 0,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn get_diag_mut(&mut self) -> DiagViewMut<&mut T, T, DynSquare> {
        DiagViewMut {
            ptr: self.buf.as_mut_ptr(),
            pos: 0,
            rank: self.rank,
            _ref_type: PhantomData,
        }
    }

    pub fn push_final_col_iter<DerefT: Deref<Target = T>, Itr: Iterator<Item = DerefT>>(
        &mut self,
        iter: Itr,
    ) {
        let DynSquare(rank) = self.rank;
        let new = rank + 1;

        let default = T::zero();
        let new_iter = iter.map(|x| *x).chain(repeat(default)).take(new);

        self.buf.reserve(new); //Could probably bypass this if we can assure that new iter has trusted_len
        self.buf.extend(new_iter);
        self.rank.grow();
    }

    pub fn push_final_col_iter_owned<Itr: Iterator<Item = T>>(&mut self, iter: Itr) {
        let new = self.rank.to_usize() + 1;

        let default = T::zero();
        let new_iter = iter.map(|x| x).chain(repeat(default)).take(new);

        self.buf.reserve(new); //Could probably bypass this if we can assure that new iter has trusted_len
        self.buf.extend(new_iter);
        self.rank.grow();
    }

    pub fn push_final_col(&mut self, vec: &[T]) {
        self.push_final_col_iter(vec.iter())
    }

    pub fn map<B, F: FnMut(&T) -> B>(&self, f: F) -> UpperTriRawData<B, DynSquare> {
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
        let mut return_vec = Vec::with_capacity(self.rank.0);
        let col_offset = index * (index + 1) / 2;
        let col_end = col_offset + index + 1;
        let tmp = self.buf.drain(Range {
            start: col_offset,
            end: col_end,
        });

        return_vec.extend(tmp);
        let mut removed = index;
        for col in (index + 1)..self.rank.0 {
            let next_to_remove = offset_for_col(col, index) - removed;
            let t = self.buf.remove(next_to_remove);
            return_vec.push(t);
            removed += 1;
        }
        self.rank.shrink();
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

impl<'a, T> Add<&'a UpperTriRawData<T, DynSquare>> for &'a UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T, DynSquare>;

    fn add(self, rhs: &'a UpperTriRawData<T, DynSquare>) -> Self::Output {
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

impl<'a, T> SubAssign<&'a UpperTriRawData<T, DynSquare>> for UpperTriRawData<T, DynSquare>
where
    T: for<'b> SubAssign<&'b T> + Copy + Zero,
{
    fn sub_assign(&mut self, rhs: &'a UpperTriRawData<T, DynSquare>) {
        self.iter_mut()
            .zip(rhs.iter())
            .for_each(|(left, right)| *left -= right);
    }
}

impl<T> SubAssign<UpperTriRawData<T, DynSquare>> for UpperTriRawData<T, DynSquare>
where
    T: SubAssign<T> + Copy + Zero,
{
    fn sub_assign(&mut self, rhs: UpperTriRawData<T, DynSquare>) {
        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(left, right)| *left -= right);
    }
}

impl<T: Display + Zero + Copy> Display for UpperTriRawData<T, DynSquare> {
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

impl<'a, T> Add<&'a UpperTriRawData<T, DynSquare>> for UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T, DynSquare>;

    fn add(self, rhs: &'a UpperTriRawData<T, DynSquare>) -> Self::Output {
        &self + rhs
    }
}

impl<'a, T> Add<UpperTriRawData<T, DynSquare>> for &'a UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T, DynSquare>;

    fn add(self, rhs: UpperTriRawData<T, DynSquare>) -> Self::Output {
        self + &rhs
    }
}

impl<'a, T> Add<UpperTriRawData<T, DynSquare>> for UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    type Output = UpperTriRawData<T, DynSquare>;

    fn add(self, rhs: UpperTriRawData<T, DynSquare>) -> Self::Output {
        &self + &rhs
    }
}

impl<'a, T> AddAssign<&'a UpperTriRawData<T, DynSquare>> for UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    fn add_assign(&mut self, rhs: &'a UpperTriRawData<T, DynSquare>) {
        self.buf
            .iter_mut()
            .zip(rhs.buf.iter())
            .for_each(|(return_val, new)| {
                *return_val += new;
            });
    }
}

impl<T> AddAssign<UpperTriRawData<T, DynSquare>> for UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero + for<'b> AddAssign<&'b T>,
{
    fn add_assign(&mut self, rhs: UpperTriRawData<T, DynSquare>) {
        *self += &rhs;
    }
}
