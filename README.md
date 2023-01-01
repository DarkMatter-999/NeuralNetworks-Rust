# Neural Networks in Rust

This is a Neural Networks implementation in pure Rust.
The only libraries used are -

- `rand` - for generating random weights
- `libm` - for for various math functions

---

## Usage

```rust


// Make a network
let network: Vec<Box<dyn Learn>> = vec![
    Box::new(DenseLayer::new(2, 3)),
    Box::new(Sigmoid::new(3, 3)),
    Box::new(DenseLayer::new(3, 1)),
    Box::new(Sigmoid::new(1, 1)),
];

// Add inputs
let xtrain = vec2d![[....]];
let ytrain = vec2d![[....]];

// Train
let trained_network = train(network, 1000, &xtrain, &ytrain, 0.1);

// Predict
let output = predict(&trained_network, &<xtest>);


```

## TODO:

Add convolution and other activation functions
