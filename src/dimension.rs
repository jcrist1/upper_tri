/// A trait for square upper triangle matrices
pub trait SquareDimension {
    fn to_usize(&self) -> usize;
}

/// A dynamic dimension.  For matrices that will need to be resized at runtime
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct DynSquare(pub(crate) usize);

impl SquareDimension for DynSquare {
    fn to_usize(&self) -> usize {
        self.0
    }
}

impl DynSquare {
    pub(crate) fn grow(&mut self) {
        self.0 += 1;
    }

    pub(crate) fn shrink(&mut self) {
        if self.0 > 0 {
            self.0 -= 1
        }
    }
}

#[derive(Clone, Copy)]
pub struct StcSquare<const N: usize>;
