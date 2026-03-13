// File Name: matrix.rs
// File Purpose: Matrix Calculations in the most memory efficiant and fastest way possible
// File Author:  Marawan Yakout  (M.Y)
// Date Created: 2026-03-13 | 13 March 2026
// Personal Note: I always like the movie matrix :)
// Last Edited Date: 13 March 202
// Last Edited by: M.Y


    // the world we havent created wasnt created with our thinking,
    //so we can't possibly understand it without changing our thinking"

// usize:
// Your machine is 64-bit  →  usize = u64  (0 to 18,446,744,073,709,551,615)
// Your machine is 32-bit  →  usize = u32  (0 to 4,294,967,295)

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix
{
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,   // Single allocation - data[row][col]
}


//  CONSTRUCTORS  (associated functions — called as Matrix::x())

impl Matrix
{
    // Create a Matrix using Vec<f64>
    // Example:  2x3 matrix :
    //                       [1,2,3]
    //                       [4,5,6]
    // Use -> let m = Matrix::new(2,3, vec![1.0, 2,0, 3.0, 4.0 , 5,0, 6.0]).unwrap();
    // Author Comment : We can also use the following
    // let m = Matrix::new(2,3, [1,2,3,4,5,6]); [TESTING]

    pub fn new<I,T>(rows: usize, col: usize, data: I) -> result<Self, Error>
    where
        I : IntoIterator<Item = T>
        T : Intro<f64>
    {
        // Convert input into (array,vec) into Vec<f64>
        // This line allows the use on integers inside the matrix (testing)
        let data: Vec<f64> = data.into_iter().map(Into:into).collect();

        if rows == 0 || cols = 0
        {
            return Err(Error::InvalidInput(
                "Wrong Dimensions, Can't be Zero".into(),
            ));
        }



        if data.len() != rows * cols
        {
            return Err(Error::InvalidInput(Format!(
                "Expected {} elements for {}x{} matrix, got {}",
                rows * cols, rows, cols, data.len()
            )));
        }

        Ok(Matrix { rows, cols, data })

    }


    //Creates a matrix that is filled with Zeros
    pub fn zeros(rows: usize, cols: usize) -> self
        {
            Matrix { rows,
                     cols,
                     data: vec![0.0; rows * cols]
            }
        }

    //Creates a matrix that is filled with Ones
    pub fn ones(rows: usize, cols: usize) -> self
        {
            Matrix { rows,
                     cols,
                     data: vec![1.0; rows * cols]
            }
        }

    //Creates a matrix that is filled with a (Certian Value)
    pub fn fill(rows: usize, cols: usize, value: f64) -> self
        {
            Matrix { rows,
                     cols,
                     data: vec![value; rows * cols]
            }
        }


    // Create an identity matrix (1s on diagonal, 0s elsewhere)
    // Only valid for square matrices
    // Usage :
    //  identity(3):  [1, 0, 0]
    //                [0, 1, 0]
    //                [0, 0, 1]
    // let eye = Matrix::identity(3);
    pub fn identity(size: usize) -> Self {
        let mut m = Matrix::zeros(size, size);
        for i in 0..size {
            m.set(i, i, 1.0); // i.e 1x1 , 2x2 , 3x3 -> 1.0
        }
        m // return matrix
    }

     /// Create from a 2D Vec (rows of Vecs) — convenience constructor
    pub fn form_2d(data: Vec<Vec<f64>>) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::InvalidInput("Matrix data is empty | Matrix cannot be empty".into()));
        }
        let rows = data.len();
        let cols = data[0].len();
        if data.iter().any(|row| row.len() != cols) {
            return Err(Error::InvalidInput(
                "Rows number must match that of columns".into(),
            ));
        }
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        Ok(Matrix { rows, cols, data: flat })
    }


}


impl Matrix
{


    pub fn dot(&self, other: &Matrix) -> Result<Matrix, crate::error::Error>
    {

        //Matrix Shape Check
        if self.cols != other.rows
        {
            return Err(crate::error::Error::ShapeMismatch {
                expected: self.cols;
                got: other.rows,
            // format!("Expected {} elements, got {}", rows * cols, data.len())
        });
        }

        //mut is a value that can be modified
        let mut result = Matrix::zeros(self)


    }

}
