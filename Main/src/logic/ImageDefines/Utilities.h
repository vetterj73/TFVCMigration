/*
	The collection of utlilty functions
*/

#pragma once


void inverse(	
	const double* inMatrix,
	double* outMatrix,
	unsigned int rows,
	unsigned int cols);

bool MultiProjTrans(
	const double leftTrans[3][3], 
	const double rightTrans[3][3], 
	double outTrans[3][3]);

