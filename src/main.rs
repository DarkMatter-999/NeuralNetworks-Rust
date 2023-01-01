use NeuralNet::{dense::DenseLayer, layer::Learn, sigmoid::Sigmoid, train};

// Macro to make 2d vector from array
// vec![vec![T], vec![T]...]
macro_rules!vec2d {
    [ $( [ $( $d:expr ),* ] ),* ] => {
        vec![
            $(
                vec![$($d),*],
            )*
        ]
    }
}

fn main() {
    let mut network: Vec<Box<dyn Learn>> = vec![
        Box::new(DenseLayer::new(2, 3)),
        Box::new(Sigmoid::new(3, 3)),
        Box::new(DenseLayer::new(3, 1)),
        Box::new(Sigmoid::new(1, 1)),
    ];

    let xtrain = vec2d![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ytrain = vec2d![[0.0], [1.0], [1.0], [0.0]];

    let epochs = 10000;
    let learning_rate = 1.0;

    let trainedNetword = train(network, epochs, &xtrain, &ytrain, learning_rate);
}
