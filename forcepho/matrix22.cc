/* matrix22.cc
This creates a simple class for 2x2 real-valued matrices.
Convention: 
    v11 v12
    v21 v22

TODO: We should template this.
*/



/* Testing if we don't need this:
// Apparently binary operator overloads need to be outside the class
class matrix22;

// matrix * scalar (and the reverse)
CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const float s);
CUDA_CALLABLE_MEMBER inline matrix22 operator * (const float s, const matrix22& A);

/// matrix + matrix, matrix - matrix
CUDA_CALLABLE_MEMBER inline matrix22 operator + (const matrix22& A, const matrix22& B);
CUDA_CALLABLE_MEMBER inline matrix22 operator - (const matrix22& A, const matrix22& B);

/// matrix * matrix product
CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const matrix22& B);
*/


	


class matrix22 {
  public:
    float v11, v12, v21, v22;
    CUDA_CALLABLE_MEMBER ~matrix22() { }     // A null destructor

    // --------------- Constructors -------------------
    CUDA_CALLABLE_MEMBER matrix22() { }    // A null constructor

    /// Filling in all the values
    CUDA_CALLABLE_MEMBER matrix22(float _v11, float _v12, float _v21, float _v22) {
      v11 = _v11; v12 = _v12; v21 = _v21; v22 = _v22; }

    /// Filling in just the diagonals
    CUDA_CALLABLE_MEMBER matrix22(float _d1, float _d2) {
        v11 = _d1; v12 = v21 = 0.0; v22 = _d2; }

    /// Read from 4 consecutive floats
    CUDA_CALLABLE_MEMBER matrix22(const float d[]) {
        v11 = d[0]; v12 = d[1]; v21 = d[2]; v22 = d[3];
    }
	
	CUDA_CALLABLE_MEMBER inline void debug_print(){
		printf(" %f %f\n", v11, v12);
		printf(" %f %f\n\n", v21, v22);
	}

    // --------------------------------------------------
    // The following operations are almost constructors, but require too much
    // attention.  Instead call as:
    //     matrix22 m; m.rot(M_PI/4.0);

    /// The identity matrix
    CUDA_CALLABLE_MEMBER inline void eye() { v11 = v22 = 1.0; v12 = v21 = 0.0; }

    /// The zero matrix
    CUDA_CALLABLE_MEMBER inline void zero() { v11 = v12 = v21 = v22 = 0.0; }

    /// A rotation matrix, with input angle in radians
    CUDA_CALLABLE_MEMBER inline void rot(float theta) {
        float c = cos(theta); float s = sin(theta);
        v11 = v22 = c;  v12 = -s; v21 = s;
    }
    
    CUDA_CALLABLE_MEMBER inline void rotation_matrix_deriv(float theta){
        float c = cos(theta); float s = sin(theta);
        v11 = v22 = -s; v12 = -c; v21 = c; 
        // return matrix22( -s, -c, c, -s);
    }
    
    CUDA_CALLABLE_MEMBER inline void scale(float q) {
        v11 = 1.0/q; v22 = q; v12 = v21 = 0.0;
    }
    
    CUDA_CALLABLE_MEMBER inline void scale_matrix_deriv(float q){
        v11 = -1.0/(q*q); v22 = 1.0; v12 = v21 = 0.0;
        // return matrix22(-1.0/(q*q), 1.0);
    }

    // --------------------------------------------------
    // The following operations return a new output matrix
    // The input instance is not altered
    
    /// The transpose
    CUDA_CALLABLE_MEMBER inline matrix22 T() { return matrix22(v11, v21, v12, v22); }

    /// The determinant
    CUDA_CALLABLE_MEMBER inline float det() { return v11*v22-v12*v21; }

    /// The trace
    CUDA_CALLABLE_MEMBER inline float trace() { return v11+v22; }

    /// The inverse
    CUDA_CALLABLE_MEMBER inline matrix22 inv() {
        float idet = 1.0/this->det();
        return matrix22(v22*idet, -v12*idet, -v21*idet, v11*idet);
    }

    // matrix *= scalar 
    CUDA_CALLABLE_MEMBER inline matrix22& operator *= ( const float s) {
        v11*=s; v12*=s; v21*=s; v22*=s; return *this;
    }

};


// Apparently binary operator overloads need to be outside the class

// matrix * scalar (and the reverse)
CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const float s) {
    return matrix22(A.v11*s, A.v12*s, A.v21*s, A.v22*s);
}
CUDA_CALLABLE_MEMBER inline matrix22 operator * (const float s, const matrix22& A) {
    return matrix22(s*A.v11, s*A.v12, s*A.v21, s*A.v22);
}

/// matrix + matrix, matrix - matrix
CUDA_CALLABLE_MEMBER inline matrix22 operator + (const matrix22& A, const matrix22& B) {
    return matrix22( A.v11+B.v11, A.v12+B.v12, A.v21+B.v21, A.v22+B.v22); 
}
CUDA_CALLABLE_MEMBER inline matrix22 operator - (const matrix22& A, const matrix22& B) {
    return matrix22( A.v11-B.v11, A.v12-B.v12, A.v21-B.v21, A.v22-B.v22); 
}

/// matrix * matrix product
CUDA_CALLABLE_MEMBER inline matrix22 operator * (const matrix22& A, const matrix22& B) {
    return matrix22( A.v11*B.v11+A.v12*B.v21, A.v11*B.v12+A.v12*B.v22,
                A.v21*B.v11+A.v22*B.v21, A.v21*B.v12+A.v22*B.v22);
}

// ========== And here are functions that work on matrices ===========

///Compute A B A, return matrix
CUDA_CALLABLE_MEMBER inline matrix22 ABA(const matrix22& A, matrix22& B){ //mamma mia! 
	float v11 = A.v11 * A.v11 * B.v11 + A.v11 * A.v21 * B.v12 + A.v11 * A.v12 * B.v21 + A.v12 * A.v21 * B.v22; 	
    float v12 = A.v11 * A.v12 * B.v11 + A.v11 * A.v22 * B.v12 + A.v12 * A.v12 * B.v21 + A.v12 * A.v22 * B.v22; 	
    float v21 = A.v11 * A.v21 * B.v11 + A.v21 * A.v21 * B.v12 + A.v11 * A.v22 * B.v21 + A.v21 * A.v22 * B.v22; 	
    float v22 = A.v12 * A.v21 * B.v11 + A.v21 * A.v22 * B.v12 + A.v12 * A.v22 * B.v21 + A.v22 * A.v22 * B.v22; 
    return matrix22(v11, v12, v21, v22); 
}

/// Compute A A^T, return the symmetrix matrix
CUDA_CALLABLE_MEMBER inline matrix22 AAt(const matrix22& A) {
    float tmp = A.v11*A.v21+A.v12*A.v22;
    return matrix22(A.v11*A.v11+A.v12*A.v12, tmp, tmp, A.v21*A.v21+A.v22*A.v22);
}
/// Compute A^T A (not the same as above!)
CUDA_CALLABLE_MEMBER inline matrix22 AtA(const matrix22& A) {
    float tmp = A.v11*A.v12+A.v21*A.v22;
    return matrix22(A.v11*A.v11+A.v21*A.v21, tmp, tmp, A.v12*A.v12+A.v22*A.v22);
}

/// Compute A B^T + B A^T.  This is symmetric and can be computed faster.
CUDA_CALLABLE_MEMBER inline matrix22 symABt(const matrix22& A, const matrix22& B) {
    float tmp = A.v11*B.v21 + A.v12*B.v22 + A.v21*B.v11 + A.v22*B.v12;
    return matrix22( 2.0*(A.v11*B.v11+A.v12*B.v12), tmp, tmp, 2.0*(A.v21*B.v21+A.v22*B.v22));
}


/// matrix^T matrix matrix triple symmetric product
/// This assumes A is symmetric!
CUDA_CALLABLE_MEMBER inline matrix22 BtAB(const matrix22& A, const matrix22& B) {
    matrix22 C = A*B;
    float tmp = B.v11*C.v12+B.v21*C.v22;
    return matrix22( B.v11*C.v11+B.v21*C.v21, tmp, tmp, B.v12*C.v12+B.v22*C.v22);
}
CUDA_CALLABLE_MEMBER inline matrix22 BABt(const matrix22& A, const matrix22& B) {
    matrix22 C = B*A;
    float tmp = C.v11*B.v21+C.v12*B.v22;
    return matrix22( C.v11*B.v11+C.v12*B.v12, tmp, tmp, C.v21*B.v21+C.v22*B.v22);
}

/// A vector*matrix*vector compression to a float
CUDA_CALLABLE_MEMBER inline float vtAv(matrix22& A, float v1, float v2) {
    return v1*v1*A.v11 + v1*v2*(A.v12+A.v21) + v2*v2*A.v22;
}

/// A matrix * vector product, returning in place
CUDA_CALLABLE_MEMBER inline void Av(matrix22& A, float *v) {
    float v1 = v[0]; float v2 = v[1];
    v[0] = A.v11 * v1 + A.v21 * v2; 
    v[1] = A.v12 * v1 + A.v22 * v2; 
}

/// A matrix * vector product, returning out of place
CUDA_CALLABLE_MEMBER inline void Av(float *w, matrix22& A, float *v) {
    float v1 = v[0]; float v2 = v[1];
    w[0] = A.v11 * v1 + A.v21 * v2; 
    w[1] = A.v12 * v1 + A.v22 * v2; 
}


