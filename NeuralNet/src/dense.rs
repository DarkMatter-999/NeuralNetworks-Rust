use crate::{
    layer::{Layer, Learn},
    matrix::Matrix,
    matrixop::{add, dot, scale, subtract, transpose},
};

pub struct DenseLayer {
    pub layer: Layer,
    pub weights: Matrix,
    pub biases: Matrix,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> DenseLayer {
        let mut weights = Matrix::new(output_size, input_size);
        let mut biases = Matrix::new(output_size, 1);
        weights.randomize();
        biases.randomize();

        DenseLayer {
            layer: Layer::new(input_size, output_size),
            weights: weights,
            biases: biases,
        }
    }
}

impl Learn for DenseLayer {
    fn forward(&mut self, inputs: Matrix) -> Matrix {
        self.layer.inputs = inputs;

        // println!("inputs {:?}", self.layer.inputs.data);

        let mut out = dot(&self.weights, &self.layer.inputs).unwrap();

        out = add(&out, &self.biases).unwrap();

        out
    }

    fn backward(&mut self, output_gradient: Matrix, learning_rate: f64) -> Matrix {
        let weights_gradient = dot(&output_gradient, &transpose(&self.layer.inputs)).unwrap();

        let input_gradient = dot(&transpose(&self.weights), &output_gradient).unwrap();

        self.weights = subtract(&self.weights, &scale(&weights_gradient, learning_rate)).unwrap();
        self.biases = subtract(&self.biases, &scale(&output_gradient, learning_rate)).unwrap();

        input_gradient
    }
}
