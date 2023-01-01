use crate::{
    layer::{Layer, Learn},
    matrix::Matrix,
    matrixop::{apply, multiply},
};

pub struct Activation {
    pub layer: Layer,
    pub activation: fn(f64) -> f64,
    pub activation_prime: fn(f64) -> f64,
}

impl Activation {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: fn(f64) -> f64,
        activation_prime: fn(f64) -> f64,
    ) -> Activation {
        Activation {
            layer: Layer::new(input_size, output_size),
            activation: activation,
            activation_prime: activation_prime,
        }
    }
}

impl Learn for Activation {
    fn forward(&mut self, inputs: Matrix) -> Matrix {
        self.layer.inputs = inputs;
        let out = apply(&self.layer.inputs, self.activation);
        out
    }

    fn backward(&mut self, output_gradient: Matrix, learning_rate: f64) -> Matrix {
        let out = multiply(
            &output_gradient,
            &apply(&self.layer.inputs, self.activation_prime),
        )
        .unwrap();

        out
    }
}
