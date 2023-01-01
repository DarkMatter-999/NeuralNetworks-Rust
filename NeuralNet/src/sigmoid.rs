use libm;

use crate::{activation::Activation, layer::Learn, matrix::Matrix};

pub struct Sigmoid {
    pub activation: Activation,
}

impl Sigmoid {
    pub fn new(input_size: usize, output_size: usize) -> Sigmoid {
        fn sigmoid(x: f64) -> f64 {
            1.0 / (1.0 + libm::exp(-x))
        }
        fn sigmoid_prime(x: f64) -> f64 {
            let s = sigmoid(x);
            s * (1.0 - s)
        }

        Sigmoid {
            activation: Activation::new(input_size, output_size, sigmoid, sigmoid_prime),
        }
    }
}

impl Learn for Sigmoid {
    fn forward(&mut self, inputs: Matrix) -> Matrix {
        self.activation.forward(inputs)
    }
    fn backward(&mut self, output_gradient: Matrix, learning_rate: f64) -> Matrix {
        self.activation.backward(output_gradient, learning_rate)
    }
}
