pub mod activation;
pub mod dense;
pub mod layer;
pub mod matrix;
pub mod matrixop;
pub mod mse;
pub mod sigmoid;

use crate::{
    layer::Learn,
    matrix::Matrix,
    mse::{mse, mse_prime},
};

pub fn train(
    mut network: Vec<Box<dyn Learn>>,
    epochs: u64,
    xtrain: &Vec<Vec<f64>>,
    ytrain: &Vec<Vec<f64>>,
    learning_rate: f64,
) -> Vec<Box<dyn Learn>> {
    let mut error = 0.0;

    for e in 0..epochs {
        for (x, y) in xtrain.into_iter().zip(ytrain.into_iter()) {
            // forward
            let mut output = Matrix::new(2, 1);
            for i in 0..output.rows {
                output.data[i][0] = x[i];
            }

            let mut y_ = Matrix::new(1, 1);
            y_.data[0][0] = y[0];

            for layer in &mut network {
                output = layer.forward(output);
            }

            // println!("{:?} => {:?} / {:?}", x, y, &output.data);

            // error
            error += mse(&y_, &output);

            //backward
            let mut grad = mse_prime(&y_, &output);

            // println!("grad: {:?}", grad.data);

            network.reverse();
            for layer in &mut network {
                grad = layer.backward(grad, learning_rate);
            }
            network.reverse();
        }

        error /= xtrain.len() as f64;
        println!("{:}/{:} error={:}", e + 1, epochs, error);
    }

    network
}
