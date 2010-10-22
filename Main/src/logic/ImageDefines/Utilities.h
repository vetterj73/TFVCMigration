/*
	The collection of utlilty functions
*/

#pragma once

// Inverse a matrix,
// inMatrix: input matrix, data stored row by row
// outMatrix: output Matrix, data stored row by row
// rows and cols: size of matrix 
void inverse(	
	const double* inMatrix,
	double* outMatrix,
	unsigned int rows,
	unsigned int cols);