use libm;

use crate::{activation::Activation, layer::Learn, matrix::Matrix};

pub struct Tanh {
    pub activation: Activation,
}

impl Tanh {
    pub fn new(input_size: usize, output_size: usize) -> Tanh {
        fn tanh(input: f64) -> f64 {
            libm::tanh(input)
        }

        fn tanh_prime(input: f64) -> f64 {
            1.0 - libm::tanh(input) * libm::tanh(input)
        }

        Tanh {
            activation: Activation::new(input_size, output_size, tanh, tanh_prime),
        }
    }
}

impl Learn for Tanh {
    fn forward(&mut self, inputs: Matrix) -> Matrix {
        self.activation.forward(inputs)
    }

    fn backward(&mut self, output_gradient: Matrix, learning_rate: f64) -> Matrix {
        let out = self.activation.backward(output_gradient, learning_rate);
        out
    }
}
