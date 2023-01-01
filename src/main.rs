use NeuralNet::{matrix::Matrix, matrixop::dot};

fn main() {
    let mut mat = Matrix::new(4, 4);
    mat.randomize();

    mat.print();

    let mut mat2 = Matrix::new(4, 4);
    mat2.randomize();

    mat2.print();

    let mat3 = dot(&mat, &mat2).unwrap();

    mat3.print()
}
