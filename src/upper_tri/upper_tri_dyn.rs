use crate::upper_tri::*;

impl<T> UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero,
{
    pub fn new(rank: usize) -> Self {
        let def = T::zero();
        let buf = repeat(def).take(rank * (rank + 1) / 2).collect();
        Self {
            buf,
            rank: DynSquare(rank),
        }
    }

    pub fn push_final_col_iter<DerefT: Deref<Target = T>, Itr: Iterator<Item = DerefT>>(
        &mut self,
        iter: Itr,
    ) {
        let rank = self.rank.to_usize();
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

    pub fn drop_at(&mut self, index: usize) -> Vec<T> {
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
}

impl<'a, 'b, T> Add<&'b UpperTriRawData<T, DynSquare>> for &'a UpperTriRawData<T, DynSquare>
where
    T: Copy + Zero + for<'c> AddAssign<&'c T>,
{
    type Output = UpperTriRawData<T, DynSquare>;

    fn add(self, rhs: &'b UpperTriRawData<T, DynSquare>) -> Self::Output {
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
