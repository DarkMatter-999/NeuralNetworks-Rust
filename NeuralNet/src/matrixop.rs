use crate::matrix::Matrix;

fn check_dimensions(m1: &Matrix, m2: &Matrix) -> bool {
    m1.rows == m2.rows && m1.columns == m2.columns
}

pub fn add(m1: &Matrix, m2: &Matrix) -> Result<Matrix, ()> {
    if check_dimensions(&m1, &m2) {
        let mut mat = Matrix::new(m1.rows, m1.columns);

        for i in 0..mat.rows {
            for j in 0..mat.columns {
                mat.data[i][j] = &m1.data[i][j] + &m2.data[i][j];
            }
        }

        Ok(mat)
    } else {
        return Err(());
    }
}

pub fn subtract(m1: &Matrix, m2: &Matrix) -> Result<Matrix, ()> {
    if check_dimensions(&m1, &m2) {
        let mut mat = Matrix::new(m1.rows, m1.columns);

        for i in 0..mat.rows {
            for j in 0..mat.columns {
                mat.data[i][j] = &m1.data[i][j] - &m2.data[i][j];
            }
        }

        Ok(mat)
    } else {
        return Err(());
    }
}

pub fn multiply(m1: &Matrix, m2: &Matrix) -> Result<Matrix, ()> {
    if check_dimensions(&m1, &m2) {
        let mut mat = Matrix::new(m1.rows, m1.columns);

        for i in 0..mat.rows {
            for j in 0..mat.columns {
                mat.data[i][j] = &m1.data[i][j] * &m2.data[i][j];
            }
        }

        Ok(mat)
    } else {
        return Err(());
    }
}

pub fn dot(m1: &Matrix, m2: &Matrix) -> Result<Matrix, ()> {
    if m1.columns == m2.rows {
        let mut mat = Matrix::new(m1.rows, m2.columns);

        for i in 0..m1.rows {
            for j in 0..m2.columns {
                let mut sum = 0.0;
                for k in 0..m2.rows {
                    sum += m1.data[i][k] * m2.data[k][j];
                }
                mat.data[i][j] = sum;
            }
        }

        Ok(mat)
    } else {
        return Err(());
    }
}

pub fn scale(m1: &Matrix, factor: f64) -> Matrix {
    let mut mat = m1.clone();
    for i in 0..mat.rows {
        for j in 0..mat.columns {
            mat.data[i][j] *= factor;
        }
    }

    mat
}

pub fn add_scalar(m1: &Matrix, factor: f64) -> Matrix {
    let mut mat = m1.clone();
    for i in 0..mat.rows {
        for j in 0..mat.columns {
            mat.data[i][j] += factor;
        }
    }

    mat
}

pub fn transpose(m1: &Matrix) -> Matrix {
    let mut mat = Matrix::new(m1.columns, m1.rows);
    for i in 0..mat.rows {
        for j in 0..mat.columns {
            mat.data[i][j] = m1.data[j][i];
        }
    }

    mat
}

pub fn apply(m1: &Matrix, func: fn(f64) -> f64) -> Matrix {
    let mut mat = m1.clone();
    for i in 0..mat.rows {
        for j in 0..mat.columns {
            mat.data[i][j] = (func)(m1.data[i][j]);
        }
    }

    mat
}
