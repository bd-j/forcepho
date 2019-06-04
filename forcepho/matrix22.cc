/* matrix22.cc
This creates a simple class for 2x2 real-valued matrices.
Convention: 
    v11 v12
    v21 v22

TODO: We should template this.
*/

class matrix22 {
  public:
    float v11, v12, v21, v22;

    matrix22() { }    // A null constructor
    /// Filling in all the values
    matrix22(float _v11, float _v12, float _v21, float _v22) {
      v11 = _v11; v12 = _v12; v21 = _v21; v22 = _v22; }

    /// Filling in just the diagonals
    matrix22(float _d1, float _d2) {
        v11 = _d1; v12 = v21 = 0.0; v22 = _d2; }
  
    ~matrix22() { }     // A null destructor

    /// The identity matrix
    inline matrix22 eye() { return matrix22(1.0, 1.0); }

    /// The zero matrix
    inline matrix22 zero() { return matrix22(0.0, 0.0); }

    /// A rotation matrix, with input angle in radians
    inline matrix22 rot(float theta) {
        float c = cos(theta); float s = sin(theta);
        return matrix22(c,-s,s,c);
    }
    
    inline matrix22 scale(float q) {
        return matrix22(1.0/q, q);
    }
    
    inline matrix22 scale_matrix_deriv(float q){
        return matrix22(-1.0/(q*q), 1.0);
    }
    
    inline matrix22 rotation_matrix_deriv(float theta){
        float c = cos(theta); float s = sin(theta);
        return matrix22( -s, -c, c, -s);
    }
    
    /// The transpose
    inline matrix22 T(matrix22& A) { return matrix22(A.v11, A.v21, A.v12, A.v22); }

    /// The determinant
    inline float det(matrix22& A) { return A.v11*A.v22-A.v12*A.v21; }

    /// The trace
    inline float trace(matrix22& A) { return A.v11+A.v22; }

    /// The inverse
    inline matrix22 inv(matrix22& A) {
        float idet = 1.0/det(A);
        return matrix22(A.v22*idet, -A.v12*idet, -A.v21*idet, A.v11*idet);
    }

    // matrix * scalar
    inline matrix22& operator * (const matrix22& A, const float s) {
        return matrix22(A.v11*s, A.v12*s, A.v21*s, A.v22*s);
    }
    inline matrix22& operator * (const float s, const matrix22& A) {
        return matrix22(s*A.v11, s*A.v12, s*A.v21, s*A.v22);
    }

    /// matrix + matrix, matrix - matrix
    inline matrix22 operator + (const matrix22& A, const matrix22& B) {
        return matrix22( A.v11+B.v11, A.v12+B.v12, A.v21+B.v21, A.v22+B.v22); 
    }
    inline matrix22 operator - (const matrix22& A, const matrix22& B) {
        return matrix22( A.v11-B.v11, A.v12-B.v12, A.v21-B.v21, A.v22-B.v22); 
    }

    /// matrix * matrix product
    inline matrix22 operator * (const matrix22& A, const matrix22& B) {
        return matrix22( A.v11*B.v11+A.v12*B.v21, A.v11*B.v12+A.v12*B.v22,
                        A.v21*B.v11+A.v22*B.v21, A.v21*B.v12+A.v22*B.v22);
    }

    /// matrix^T matrix matrix triple symmetric product
    /// This assumes A is symmetric!
    inline matrix22 BtAB(const matrix22& A, const matrix& B) {
        matrix22 C = A*B;
        float tmp = B.v11*C.v12+B.v21*C.v22,
        return matrix22( B.v11*C.v11+B.v21*C.v21, tmp, tmp, B.v12*C.v12+B.v22*C.v22);
    }
    inline matrix22 BABt(const matrix22& A, const matrix& B) {
        matrix22 C = B*A;
        float tmp = C.v11*B.v21+C.v12*B.v22,
        return matrix22( C.v11*B.v11+C.v12*B.v12, tmp, tmp, C.v21*B.v21+C.v22*B.v22);
    }

    /// A vector*matrix*vector compression to a float
    inline float vtAv(matrix22& A, float v1, float v2) {
        return v1*v1*A.v11 + v1*v2*(A.v12+A.v21) + v2*v2*A.v22;
    }

    /// We are not currently providing a matrix * vector product
    /// Note that we are not providing *= either!
};
