#include "Utilities.h"
#include "lsqrpoly.h" 

// Inverse a matrix,
// inMatrix: input matrix, data stored row by row
// outMatrix: output Matrix, data stored row by row
// rows and cols: size of matrix 
void inverse(	
	const double* inMatrix,
	double* outMatrix,
	unsigned int rows,
	unsigned int cols)
{
	double* in = new double[rows*cols];
	
	// Transpose matrix
	int ix, iy;
	for(iy=0; iy<rows; iy++)
		for(ix=0; ix<cols; ix++)
			in[ix*rows+iy] = inMatrix[iy*cols+ix];

	double*	sigma = new double[cols];
	double* answer = new double[cols*rows];
	double* b = new double [rows];

	// perform factorization of system A = QR
	int qrdRet = 
		qrd(
				in,			/* System matrix, m-by-n */
				sigma,      /* Diagonals of R (caller reserves n elements) */
				rows,     /* Number of rows in system matrix */
				rows,		/* Spacing between columns in system matrix */
				cols );	/* Number of columns in system matrix */

                            /* Return values

                                 0 - Normal completion.
                                 1 - Matrix was of incompatible dimensions.
                                 2 - Singular system matrix. */
   double condition =
	   rcond(               /* Reciprocal condition number estimator */
			   cols,		/* Number of unknowns */
			   in,			/* QR factorization returned from qrd() */
			   rows,       /* Spacing between columns of qr[] */
			   sigma        /* R diagonals from qrd() */
            );

   	for(unsigned int iter(0); iter<rows; ++iter)
	{
		for(unsigned int bIndex(0); bIndex<rows; ++bIndex)
			b[bIndex] = 0;

		b[iter] = 1;

		qrsolv (
					in,			/* Factored system matrix, from qrd() */
					sigma,		/* Diagonal of R, from qrd() */
					b,			/* Constant vector, overwritten by solution */
					rows,		/* Number of rows in system matrix */
					rows,		/* Spacing between columns in system matrix */
					cols,		/* Number of columns in system matrix */
					1     );	/*	0 - premultiply  b  by Q
									1 - premultiply  b  by inv(QR)
									2 - premultiply  b  by QR
									3 - premultiply  b  by inv(R^T R)
									4 - premultiply  b  by Q inv(R^T)
									5 - premultiply  b  by R
									6 - premultiply  b  by inv(R)

									 qrsolv() always returns 0 */

		// copy result into the answer
		for(unsigned int index2(0); index2<rows; ++index2)
			answer[iter*rows+index2] = b[index2];
	}

	// Tranpose matrix
	for(iy=0; iy<rows; iy++)
		for(ix=0; ix<cols; ix++)
			outMatrix[ix*rows+iy] = in[iy*cols+ix];

	delete [] in;
	delete [] sigma;
	delete [] answer;
	delete [] b;
}	

// Multiply two projective transforms 
// outTrans= leftTrans*rightTrans, outTrans[2][2] = 1
// leftTrans and rightTrans: input transform
// outTrans: output transform
bool MultiProjTrans(
	const double leftTrans[3][3], 
	const double rightTrans[3][3], 
	double outTrans[3][3])
{
	if(leftTrans[2][2]!=1 || rightTrans[2][2]!=1)
		return(false);

	int ix, iy;
	// Multi two input transforms
	for(iy = 0; iy < 3; iy++)
	{
		for(ix = 0; ix < 3; ix++)
		{
			outTrans[iy][ix] = leftTrans[iy][0]*rightTrans[0][ix]
				+ leftTrans[iy][1]*rightTrans[1][ix]
				+ leftTrans[iy][2]*rightTrans[2][ix];
		}
	}

	if(outTrans[3][3] < 0.01) return(false);

	// Normalize the output transform
	for(iy = 0; iy < 3; iy++)
	{
		for(ix = 0; ix < 3; ix++)
		{
			outTrans[iy][ix] /= outTrans[2][2];
		}
	}

return(true);
}