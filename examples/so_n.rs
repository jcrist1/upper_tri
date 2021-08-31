#![feature(const_generics)]
#![feature(const_evaluatable_checked)]
#![allow(incomplete_features)]

use upper_tri::dimension::StcSquare;
use upper_tri::upper_tri::upper_tri_stc::*;
use upper_tri::upper_tri::*;

const ONE: usize = 1;

const fn minus_one(n: usize) -> Option<usize> {
    if n == 0 {
        None
    } else {
        Some(n - 1)
    }
}

struct SoN<const N: usize>(UpperTriRawData<f64, StcSquare<{ N - 1 }>>)
where
    StcSquare<{ N - 1 }>: Sized;

impl<const N: usize> SoN<N>
where
    StcSquare<{ N - 1 }>: Sized,
{
    fn new() -> SoN<N> {
        let data: UpperTriRawData<f64, StcSquare<{ N - 1 }>> =
            UpperTriRawData::<f64, StcSquare<{ N - 1 }>>::new();
        SoN(data)
    }
}

impl<const N: usize> SoN<N>
where
    StcSquare<{ N - 1 }>: Sized,
{
    fn get_row<'a>(&'a self, row: usize) -> Box<dyn Iterator<Item = f64> + 'a> {
        let diag = self.0.get_diag_el(row).map(|x| *x).into_iter();
        let row_iter = self.0.get_raw_row(row).map(|x| *x);
        let end = std::iter::once(0.0).chain(diag).chain(row_iter);

        if row == 0 {
            let b = std::iter::empty().chain(end);
            Box::new(b)
        } else {
            let index = row - 1;
            let col = self.0.get_raw_col(index).map(|x| -x);
            let diag = self.0.get_diag_el(index).into_iter().map(|x| -x);
            let c = col.chain(diag).chain(end);
            Box::new(c)
        }
    }

    fn get_col<'a>(&'a self, col: usize) -> Box<dyn Iterator<Item = f64> + 'a> {
        Box::new(self.get_row(col).map(|x| -x))
    }
}

fn lie_prod<'a, const N: usize>(a: &'a SoN<N>, b: &'a SoN<N>) -> SoN<N>
where
    StcSquare<{ N - 1 }>: Sized,
{
    let iter = (0..N).flat_map(|j| {
        (0..j).map(move |i| {
            let left: f64 = a.get_row(i).zip(b.get_col(j)).map(|(x, y)| x * y).sum();
            let right: f64 = b.get_row(i).zip(a.get_col(j)).map(|(x, y)| x * y).sum();
            left - right
        })
    });

    let mut data_c = UpperTriRawData::<f64, StcSquare<{ N - 1 }>>::new();
    data_c.iter_mut().zip(iter).for_each(|(c, new)| *c = new);
    SoN(data_c)
}

macro_rules! l {
    ($left:expr, $right: expr) => {
        lie_prod($left, $right)
    };
}

fn main() {
    let mut left_data = SoN::<3>::new();
    let mut right_data = SoN::<3>::new();

    let mut output_should = SoN::<3>::new();
    *left_data.0.get_mut(0, 0).unwrap() = 1.0;
    *right_data.0.get_mut(1, 1).unwrap() = 1.0;

    println!("{}", left_data.0);
    println!("{}", right_data.0);
    *output_should.0.get_mut(0, 1).unwrap() = 1.0;
    let output = l![&left_data, &right_data];
    println!("{}", output.0);
    println!("{}", output_should.0);
    output_should.0 -= &output.0;
    let total: f64 = output_should.0.iter().map(|x| x.abs()).sum();
    println!("Total: {}", total);
    assert!(total <= 1.0e-7);
}
