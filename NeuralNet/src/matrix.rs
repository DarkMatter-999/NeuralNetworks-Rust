use rand::Rng;
pub struct Matrix {
    pub data: Vec<Vec<f64>>,
    pub rows: usize,
    pub columns: usize,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Matrix {
        let mut matrix = Vec::with_capacity(rows);
        for _ in 0..rows {
            let mut row = Vec::with_capacity(columns);
            for _ in 0..columns {
                row.push(0.0);
            }
            matrix.push(row);
        }

        Matrix {
            rows,
            columns,
            data: matrix,
        }
    }

    pub fn print(&self) {
        println!("[");
        for i in 0..self.rows {
            print!("[");
            for j in 0..self.columns {
                print!("{:.3} ", self.data[i][j]);
            }
            println!("]");
        }
        println!("]");
    }

    pub fn randomize(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] = rand::thread_rng().gen_range(-10.0..10.0);
            }
        }
    }

    pub fn fill(&mut self, n: f64) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] = n;
            }
        }
    }

    pub fn flatten(&self, axis: u8) -> Result<Matrix, ()> {
        let mut mat: Matrix;

        match axis {
            0 => {
                mat = Matrix::new(self.rows * self.columns, 1);
                for i in 0..self.rows {
                    for j in 0..self.columns {
                        mat.data[i * self.columns + j][0] = self.data[i][j];
                    }
                }
            }
            1 => {
                mat = Matrix::new(1, self.rows * self.columns);
                for i in 0..self.rows {
                    for j in 0..self.columns {
                        mat.data[0][i * self.columns + j] = self.data[i][j];
                    }
                }
            }
            _ => {
                return Err(());
            }
        }
        Ok(mat)
    }

    pub fn clone(&self) -> Matrix {
        Matrix {
            rows: self.rows,
            columns: self.columns,
            data: self.data.clone(),
        }
    }
}
