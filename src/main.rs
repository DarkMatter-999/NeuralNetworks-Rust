use NeuralNet::{dense::DenseLayer, layer::Learn, matrix::Matrix, matrixop::dot, sigmoid::Sigmoid};

fn main() {
    let mut network: Vec<Box<dyn Learn>> = vec![
        Box::new(DenseLayer::new(2, 3)),
        Box::new(Sigmoid::new(3, 3)),
        Box::new(DenseLayer::new(3, 1)),
        Box::new(Sigmoid::new(1, 1)),
    ];

    let x = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let y = [[0.0], [1.0], [1.0], [0.0]];

    for (x, y) in x.into_iter().zip(y.into_iter()) {
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

        println!("{:?} => {:?} / {:?}", x, y, &output.data);
    }
}
