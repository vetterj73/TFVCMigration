// RobustSolverTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "lsqrpoly.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>

using namespace std;

int ReadCSV(string fileName, double* data)
{
	 ifstream  file(fileName);     
	 if(!file.is_open()) return(-1);

	 int iCount = 0;
	 string line;     
	 while(getline(file,line))     
	 {         
		 stringstream  lineStream(line);         
		 string        cell;         
		 while(getline(lineStream,cell,','))         
		 {             // You have a cell!!!!   
			 data[iCount] = atof(cell.c_str()); //convertToDouble(cell);
			 iCount++;
		 }     
	 }

	 return(iCount);
}

int  ReadCSV(string fileName, int* data)
{
	 ifstream  file(fileName);     
	 if(!file.is_open()) return(-1);

	 int iCount = 0;
	 string line;     
	 while(getline(file,line))     
	 {         
		 stringstream  lineStream(line);         
		 string        cell;         
		 while(getline(lineStream,cell,','))         
		 {             // You have a cell!!!!   
			 data[iCount] = atoi(cell.c_str()); //convertToDouble(cell);
			 iCount++;
		 }     
	 }

	 return(iCount);
}


int _tmain(int argc, _TCHAR* argv[])
{
	string cPath = "C:\\Temp\\";

	//NOTES: The writing to disc process trunk the data a little bit
	//Therefore, data here is a little different with the original one

	int* iBlockLens = new int[5000];
	int iFlag = ReadCSV(cPath + "BlockLength.csv", iBlockLens);	
	if(iFlag<=0) 
		cout << "Read BlockLength.csv failed" << endl;
	int iCols = iBlockLens[0];
	int iRows = iBlockLens[1];	
	int iBW =  iBlockLens[2];
	
	double* dMatrixA_t = new double[iCols*iRows];	
	iFlag = ReadCSV(cPath + "MatrixA_t.csv", dMatrixA_t);
	if(iFlag<=0) 
		cout << "Read MatrixA_t.csv failed" << endl;
	double* dVectorB = new double[iRows];
	iFlag = ReadCSV(cPath + "VectorB.csv", dVectorB);
	if(iFlag<=0) 
		cout << "Read VectorB.csv failed" << endl;
	
	double scaleparm = 0;
	double cond = 0;	
	double* dVectorX = new double[iCols];
	double* dResidual = new double[iRows];

	int algHRetVal = 
		alg_hb(                // Robust regression by Huber's "Algorithm H"/ Banded version.
			// Inputs //			// See qrdb() in qrband.c for more information.
			iBW,						// Width (bandwidth) of each block 
			iCols-iBW+1,				// # of blocks 
			iBlockLens+3,						// mb[i] = # rows in block i 
			dMatrixA_t,				// System matrix
			dVectorB,			// Constant vector (not overwritten); must not coincide with x[] or resid[]. 

		   // Outputs //

			dVectorX,            // Coefficient vector (overwritten with solution) 
			dResidual,				// Residuals b - A*x (pass null if not wanted).  If non-null, must not coincide with x[] or b[].
			&scaleparm,				// Scale parameter.  This is a robust measure of
									//  dispersion that corresponds to RMS error.  (Pass NULL if not wanted.)
			&cond					// Approximate reciprocal condition number.  (Pass NULL if not wanted.) 

			// Return values
			//>=0 - normal return; value is # of iterations.
			// -1 - Incompatible dimensions (m mustn't be less than n)
			// -2 - singular system matrix
			// -3 - malloc failed.
			// -4 - Iteration limit reached (results may be tolerable)
				  );

	cout << "nIterations=" << algHRetVal 
		<< "; scaleparm=" << scaleparm 
		<< "; cond=" << cond
		<< endl;

	ofstream of(cPath + "verctox_test.csv");
	of << scientific;
	for(int i=0; i<iCols/6; i++)
	{
		for(int j=0; j<6; ++j)
		{
			if( j!=0 )
				of << ",";

			double d = dVectorX[i*6 + j];

			of << d;
		}
		of << endl;
	}
	of.close();

	delete [] iBlockLens;
	delete [] dMatrixA_t;
	delete [] dVectorB;
	delete [] dVectorX;
	delete [] dResidual;

	cout << "end of work" << endl;

	return 0;
}

