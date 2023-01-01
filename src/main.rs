use NeuralNet::matrix::Matrix;

fn main() {
    let mut mat = Matrix::new(4, 4);

    mat.print();
    mat.randomize();

    mat.print();

    mat.fill(1.0);
    mat.print();

    let mat2 = mat.flatten(0);

    match mat2 {
        Ok(mat2) => {
            mat2.print();
        }
        Err(()) => {}
    }
}
