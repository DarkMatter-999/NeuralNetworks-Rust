use crate::matrix::Matrix;

pub struct Layer {
    pub inputs: Matrix,
    pub outputs: Matrix,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        let mut inputs = Matrix::new(input_size, 1);
        let mut outputs = Matrix::new(output_size, 1);

        inputs.randomize();
        outputs.randomize();

        Layer {
            inputs: inputs,
            outputs: outputs,
        }
    }
}

pub trait Learn {
    fn forward(&mut self, inputs: Matrix) -> Matrix;

    fn backward(&mut self, output_gradient: Matrix, learning_rate: f64) -> Matrix;
}
