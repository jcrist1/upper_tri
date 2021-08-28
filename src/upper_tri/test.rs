use crate::upper_tri::*;
use std::iter::repeat;

#[test]
/// The row and column access is strictly above and to the right of the diagonal so the
/// column above position zero has length 0.  The column above position 1 has length 1 etc.
fn test_sizes() {
    let hundurd: usize = 100;
    let mut b = UpperTriRawData::<f32, DynSquare>::new(hundurd);

    let mut val = 0.0;
    let row_index = 53;
    b.get_raw_row_mut(row_index).for_each(|x| {
        *x = val;
        val += 1.0;
    });
    let c = b.get_raw_row(row_index).map(|x| *x).collect::<Vec<_>>();
    assert_eq!(
        c,
        (0..(hundurd - row_index - 1))
            .map(|x| x as f32)
            .collect::<Vec<_>>()
    );
    let d = b.get_raw_row(hundurd - 2).collect::<Vec<_>>();
    assert_eq!(d.len(), 1);
    let e = b.get_raw_col(1).collect::<Vec<_>>();
    assert_eq!(e.len(), 1);
    let row = b.get_raw_row(0).collect::<Vec<_>>();
    assert_eq!(row.len(), hundurd - 1);
    let col = b.get_raw_col(hundurd - 1).collect::<Vec<_>>();
    assert_eq!(col.len(), hundurd - 1);
}

#[test]
fn test_element_access() {
    let mut upper_tri = UpperTriRawData::<usize, DynSquare>::new(0);
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
    let mut upper_tri = UpperTriRawData::<usize, DynSquare>::new(10);
    let x: i32 = upper_tri.get_raw_col(1).map(|_| 1).sum();
    assert_eq!(x, 1); // index 1 is the second col ... arrays start at 0

    let x: i32 = upper_tri.get_raw_col(10).map(|_| 1).sum();
    assert_eq!(x, 0);

    let x: i32 = upper_tri.get_raw_col_mut(1).map(|_| 1).sum();
    assert_eq!(x, 1);

    let x: i32 = upper_tri.get_raw_col_mut(10).map(|_| 1).sum();
    assert_eq!(x, 0);

    let x: i32 = upper_tri.get_raw_row(10).map(|_| 1).sum();
    assert_eq!(x, 0);

    let x: i32 = upper_tri.get_raw_row(0).map(|_| 1).sum();
    assert_eq!(x, 9);
    let x: i32 = upper_tri.get_raw_row_mut(10).map(|_| 1).sum();
    assert_eq!(x, 0);

    let x: i32 = upper_tri.get_raw_row_mut(0).map(|_| 1).sum();
    assert_eq!(x, 9);

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
    let mut upper_tri = UpperTriRawData::<usize, DynSquare>::new(ten);
    assert_eq!(upper_tri.buf.len(), 10 * 11 / 2);
    let vec = (0..11).collect::<Vec<_>>();
    upper_tri.push_final_col(&vec);
    // dropping the "corner" at the fifth element
    // (arrays start at 0)
    upper_tri.drop_at(4);
    let col = upper_tri.get_raw_col(9).map(|x| *x).collect::<Vec<_>>();
    assert_eq!(vec![0, 1, 2, 3, 4, 6, 7, 8, 9], col);
}
#[test]
fn test_addition() {
    let ten: usize = 10;
    let mut upper_tri_1 = UpperTriRawData::<isize, DynSquare>::new(1);
    let mut upper_tri_2 = UpperTriRawData::<isize, DynSquare>::new(1);
    upper_tri_1.push_final_col_iter(repeat(&(-1)));
    upper_tri_1.push_final_col_iter(repeat(&1));
    upper_tri_2.push_final_col_iter(repeat(&1));
    upper_tri_2.push_final_col_iter(repeat(&1));

    let upper_tri_3 = upper_tri_1 + upper_tri_2;
    let second_col = upper_tri_3.get_raw_col(1).map(|x| *x).collect::<Vec<_>>();
    let third_col = upper_tri_3.get_raw_col(2).map(|x| *x).collect::<Vec<_>>();
    assert_eq!(second_col, vec![0]);
    assert_eq!(third_col, vec![2, 2]);
}
