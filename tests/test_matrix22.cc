#include "../forcepho/header.hh"

int main(){
	
	matrix22 A = matrix22(1.3, 2.12, -0.34, 4.1);
	matrix22 B = matrix22(-3.03, 1.1, -5.6, 2.3);

	
	/// Compute A B^T + B A^T.  This is symmetric and can be computed faster.
	matrix22 new_matrix = symABt(A, B); 
	new_matrix.debug_print(); 
	/// Compute A B^T + B A^T.  This is symmetric and can be computed faster.

	matrix22 C_diag = matrix22(1.2, 3.2); 
	
	C_diag.debug_print(); 
	
	float d[4]; 
	d[0] = 0.3; d[1] = -0.2; d[2] = 1.3; d[3] = 4.3; 
	matrix22 D = matrix22(d); 
	
	D.debug_print();
	
	printf("A matmul B = \n"); 
	(A*B).debug_print(); 
	
	printf("identity = \n"); 
	matrix22 I; 
	I.eye();
	I.debug_print();
	
	printf("zero = \n"); 
	matrix22 Z; 
	Z.zero();
	Z.debug_print();
	
	matrix22 R; float q = 2.1; 
	printf("scale = \n"); 
	R.scale(q);
	R.debug_print();
	
	printf("scale_matrix_deriv = \n"); 
	R.scale_matrix_deriv(q);
	R.debug_print();
	
	printf("rotation = \n"); 
	float theta = 0.3; 
	R.rot(theta);
	R.debug_print();
	
	printf("rotation_matrix_deriv = \n"); 
	R.rotation_matrix_deriv(theta);
	R.debug_print();
	
	printf("transpose of the previous matrix = \n"); 
	R = R.T();
	R.debug_print();
	
	A.debug_print();
	
	printf("determinant of the previous matrix = \n"); 
	float det = A.det();
	printf("%f\n\n", det);
	
	printf("inverse of the previous matrix = \n"); 

	A.inv().debug_print();
	
	printf("trace of the previous matrix = \n"); 
	float tr = A.inv().trace();
	printf("%f\n\n", tr);
	
	
	R = A.inv();
	
	printf("3.2 * previous matrix = \n"); 
	R = 3.2 * R; 
	(R).debug_print();
	
	printf("-1.2 * previous matrix = \n");
	R *= -1.2;  
	(R).debug_print();
	
	printf("-1.0 * previous matrix = \n");
	R = R * -1.0; 
	(R).debug_print();
	
	printf("A, B: \n");
	A.debug_print();
	B.debug_print(); 
	printf("A + B: \n");
	(A+B).debug_print();
	printf("A - B: \n");
	(A-B).debug_print();
	printf("A * B: \n");
	(A*B).debug_print();
	
	
	matrix22 aba; 
	printf("A * B * A: \n");
	aba = ABA( A, B);
	aba.debug_print();
	
	matrix22 aat; 
	printf(" A * A.T\n");
	aat = AAt(A); 
	aat.debug_print();
	
	matrix22 ata; 
	printf(" A.T * A\n");
	ata = AtA(A); 
	ata.debug_print();
	
	
	printf("making a symmetric:\n");
	A.eye(); 
	A *= -4.2; 
	A.debug_print(); 
	
	printf("B.T * A * B\n");
	matrix22 btab; 
	btab = BtAB(A, B); 
	btab.debug_print();
	
	printf("B * A * B.T\n");
	matrix22 babt; 
	babt = BABt(A, B); 
	babt.debug_print();
	
	float vector[2]; 
	vector[0] = 0.2; vector[1] = -1.45;
	printf(" v.T *A * v\n"); 
	printf("%f\n\n", vtAv(A, vector[0], vector[1]) );
	
	
	printf(" A * v\n"); 
	Av(A, vector);
	printf("%f %f \n\n", vector[0], vector[1] );
	
	float output[2]; 
	vector[0] = 0.2; vector[1] = -1.45;
	
	printf(" A * v\n"); 
	Av(output, A, vector);
	printf("%f %f \n\n", output[0], output[1] );
}
