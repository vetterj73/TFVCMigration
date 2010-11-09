#include "RobustSolver.h"


RobustSolver::RobustSolver(unsigned int iNumFovs, unsigned int iNumCorrelation, bool bProjectiveTrans)
{
	_iNumFovs = iNumFovs;
	_bProjectiveTrans = bProjectiveTrans;
	
	if(_bProjectiveTrans)
	{
		_iNumCalibConstrains = 10;
		_iNumParamsPerFov = 12;
	}
	else
	{
		_iNumCalibConstrains = 6;
		_iNumParamsPerFov = 6;
	}

	_iMatrixWidth = _iNumFovs * _iNumParamsPerFov;
	_iMatrixHeight = _iNumFovs * _iNumCalibConstrains + 2*iNumCorrelation;

	unsigned int _iMatrixSize = _iMatrixWidth * _iMatrixHeight;

	_dMatrixA = new double[_iMatrixSize];
	_dMatrixACopy = new double[_iMatrixSize];
	
	_dVectorB = new double[_iMatrixHeight];
	_dVectorBCopy = new double[_iMatrixHeight];

	_dVectorX = new double[_iMatrixWidth];
}

RobustSolver::~RobustSolver(void)
{
	delete [] _dMatrixA;
	delete [] _dMatrixACopy;
	delete [] _dVectorB;
	delete [] _dVectorBCopy;
	delete [] _dVectorX;

	ZeroTheSystem();
}

//initialize all coefficients in the system of equations to zero 
void RobustSolver::ZeroTheSystem()
{
	_iCurrentRow = 0;

	unsigned int i;
	for(i=0; i<_iMatrixSize; i++)
		_dMatrixA[i] = 0.0;

	for(i=0; i<_iMatrixHeight; i++)
	{
		_dVectorB[i] = 0.0;
	}

	for(i =0; i<_iMatrixWidth; i++)
		_dVectorX[i] = 0.0;
}

// Add Constraints for one 
void RobustSolver::AddCalibationConstraints(MosaicImage* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex)
{

}

/*/ Robust regression by Huber's "Algorithm H"
// Banded version
void RobustSolver::SolveXAlgHB()
{

	::memcpy(MatrixACopy, MatrixA, SizeOfA*sizeof(double));
	VectorBCopy = VectorB;

	// we built A in row order
	// the qr factorization method requires column order
	bool bRemoveEmptyRows = false;
	int* mb = new int[widthOfA];
	unsigned int iEmptyRows;
	unsigned int bw = ReorderAndTranspose(bRemoveEmptyRows, mb, &iEmptyRows);

	double*	resid = new double[nRows];
	double scaleparm(0);
	double cond(0);

	G_LOG_0_SPEED2("BEGIN ALG_HB");

	int algHRetVal = 
		alg_hb(                // Robust regression by Huber's "Algorithm H"/ Banded version.
			// Inputs //			// See qrdb() in qrband.c for more information.

			bw,						// Width (bandwidth) of each block 
			widthOfA-bw+1,			// # of blocks 
			mb,						// mb[i] = # rows in block i 
			MatrixA,				// System matrix
			&VectorB[0],			// Constant vector (not overwritten); must not coincide with x[] or resid[]. 

		   // Outputs //

			&VectorX[0],            // Coefficient vector (overwritten with solution) 
			&resid[0],				// Residuals b - A*x (pass null if not wanted).  If non-null, must not coincide with x[] or b[].
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

	delete [] mb;

	if( algHRetVal<0 )
		G_LOG_1_ERROR("alg_hb returned value of %d", algHRetVal);

	G_LOG_0_SPEED2("FINISHED ALG_HB");
	G_LOG_1_SOFTWARE("alg_hb Bandwidth = %d", bw)
	G_LOG_3_SOFTWARE("alg_hb nIterations=%d, scaleparm=%f, cond=%f", algHRetVal, scaleparm, cond);

	if(Config::instance().getInt(CFG_KEY_Align_Debug, CFG_VAL_Align_Debug))
	{
		std::string filename = Config::instance().getLoggingDir() + "\\MatrixA.csv";
		OutputDebugMatrixCSV(filename);

		if(_bProjeciveTrans)
		{
			filename = Config::instance().getLoggingDir() + "\\VectorX.csv";
			OutputDebugVectorCSV(filename);
		}
	}

	delete[] resid;
}*/