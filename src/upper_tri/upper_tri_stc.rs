use crate::dimension::StcSquare;
use crate::upper_tri::*;

impl<T, const N: usize> UpperTriRawData<T, StcSquare<N>>
where
    T: Copy + Zero,
{
    pub fn new() -> Self {
        let def = T::zero();
        let buf = repeat(def).take(N * (N + 1) / 2).collect();
        Self {
            buf,
            rank: StcSquare,
        }
    }
}

impl<'a, 'b, T, const N: usize> Add<&'b UpperTriRawData<T, StcSquare<N>>>
    for &'a UpperTriRawData<T, StcSquare<N>>
where
    T: Copy + for<'c> AddAssign<&'c T>,
{
    type Output = UpperTriRawData<T, StcSquare<N>>;

    fn add(self, rhs: &'b UpperTriRawData<T, StcSquare<N>>) -> Self::Output {
        let mut return_buf = rhs.buf.clone();
        return_buf
            .iter_mut()
            .zip(self.buf.iter())
            .for_each(|(return_val, new)| {
                *return_val += new;
            });

        UpperTriRawData {
            buf: return_buf,
            rank: StcSquare,
        }
    }
}
