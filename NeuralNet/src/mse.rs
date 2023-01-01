use crate::{
    matrix::Matrix,
    matrixop::{apply, scale, subtract},
};

pub fn mse(y_true: &Matrix, y_pred: &Matrix) -> f64 {
    let mut out = subtract(&y_true, &y_pred).unwrap();

    out = apply(&out, |x| x * x);

    let mut mean = 0.0;

    for i in 0..out.rows {
        mean += out.data[i][0];
    }

    mean / out.rows as f64
}

pub fn mse_prime(y_true: &Matrix, y_pred: &Matrix) -> Matrix {
    let mut out = subtract(&y_pred, &y_true).unwrap();

    let mean = 0.0;

    out = scale(&out, (2.0 / y_true.rows as f64));

    out
}
