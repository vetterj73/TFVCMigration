#include "RobustSolver.h"
#include "EquationWeights.h"

#define Weights EquationWeights::instance()

RobustSolver::RobustSolver(		
	map<FovIndex, unsigned int>* pFovOrderMap, 
	unsigned int iNumCorrelation, 
	bool bProjectiveTrans)
{
	_pFovOrderMap = pFovOrderMap;
	_bProjectiveTrans = bProjectiveTrans;	
	
	_iNumFovs = (unsigned int)pFovOrderMap->size();
	
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
bool RobustSolver::AddCalibationConstraints(MosaicImage* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex)
{
	// Validation check
	if(iCamIndex>=pMosaic->NumCameras() || iTrigIndex>=pMosaic->NumTriggers())
		return(false);

	// Fov transform parameter begin position in column
	// Fov's nominal center
	FovIndex index(pMosaic->Index(), iTrigIndex, iCamIndex); 
	int iFOVPos = (*_pFovOrderMap)[index] *_iNumParamsPerFov;
	ImgTransform transFov = pMosaic->GetImagePtr(iCamIndex, iTrigIndex)->GetNominalTransform();
	unsigned int iCols = pMosaic->GetImagePtr(iCamIndex, iTrigIndex)->Columns();
	unsigned int iRows = pMosaic->GetImagePtr(iCamIndex, iTrigIndex)->Rows();
	double dPixelCenRow = (iRows-1) / 2.0;
	double dPixelCenCol = (iCols-1) / 2.0;
	double dFovCalCenX, dFovCalCenY;
	transFov.Map(dPixelCenRow, dPixelCenCol, &dFovCalCenX, &dFovCalCenY);
	
	// Next camera Fov transform parameter begin position in column
	// Next camera Fov's nominal center
	index.CameraIndex++;
	int iNextCamFovPos = -1;
	ImgTransform transNextCamFov;
	double dNextCamFovCalCenX, dNextCamFovCalCenY;
	if(index.CameraIndex < pMosaic->NumCameras())
	{
		iNextCamFovPos = (*_pFovOrderMap)[index] * _iNumParamsPerFov;
		transNextCamFov = pMosaic->GetImagePtr(++iCamIndex, iTrigIndex)->GetNominalTransform();
		transFov.Map(dPixelCenRow, dPixelCenCol, &dNextCamFovCalCenX, &dNextCamFovCalCenY);
	}
	
	/*/ Next trigger Fov transform parameter begin position in column
	// Next trigger Fov's nominal transform
	index.CameraIndex--;
	index.TriggerIndex++;
	int iNextTrigFovPos = -1;
	ImgTransform transNextTrigFov;
	double dNextTrigFovCalCenX, dNextTrigFovCalCenY;
	if(index.TriggerIndex < pMosaic->NumTriggers())
	{
		iNextTrigFovPos = (*_pFovOrderMap)[index] * _iNumParamsPerFov;
		transNextTrigFov = pMosaic->GetImagePtr(iCamIndex, ++iTrigIndex)->GetNominalTransform();
		transFov.Map(dPixelCenRow, dPixelCenCol, &dNextTrigFovCalCenX, &dNextTrigFovCalCenY);
	}*/

	double* pdRowBegin = _dMatrixA + _iCurrentRow*_iMatrixWidth;

	//* 1 rotation match 
	pdRowBegin[iFOVPos+1] = Weights.wRxy;
	pdRowBegin[iFOVPos+3] = Weights.wRxy;
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 2  magnification match 
	pdRowBegin[iFOVPos+0] = Weights.wMxy;
	pdRowBegin[iFOVPos+4] = -Weights.wMxy;
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 3 rotate images 90 degrees 
	pdRowBegin[iFOVPos+1] = Weights.wRcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = -Weights.wRcal * transFov.GetItem(1);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 4 rotate images 90 degrees 
	pdRowBegin[iFOVPos+3] = Weights.wRcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = -Weights.wRcal * transFov.GetItem(3);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 5 square pixels x 
	pdRowBegin[iFOVPos+0] = Weights.wMcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = Weights.wMcal * transFov.GetItem(0);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 6 square pixels y 
	pdRowBegin[iFOVPos+4] = Weights.wMcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = Weights.wMcal * transFov.GetItem(4);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 7 fov center Y pos
	pdRowBegin[iFOVPos+3] = Weights.wYcent * dPixelCenRow;
	pdRowBegin[iFOVPos+4] = Weights.wYcent * dPixelCenCol;
	pdRowBegin[iFOVPos+5] = Weights.wYcent;
	if(_bProjectiveTrans)
	{
		pdRowBegin[iFOVPos+9] = Weights.wYcent * dPixelCenRow * dPixelCenRow;
		pdRowBegin[iFOVPos+10] = Weights.wYcent * dPixelCenRow * dPixelCenCol;
		pdRowBegin[iFOVPos+11] = Weights.wYcent * dPixelCenCol * dPixelCenCol;
	}
	_dVectorB[_iCurrentRow] = Weights.wYcent  * dFovCalCenY;
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 8 distance between cameras in Y
	if(iNextCamFovPos > 0)
	{
		pdRowBegin[iFOVPos+3]		= Weights.wYdelta * dPixelCenRow;
		pdRowBegin[iFOVPos+4]		= Weights.wYdelta * dPixelCenCol;
		pdRowBegin[iFOVPos+5]		= Weights.wYdelta;
		pdRowBegin[iNextCamFovPos+3]	= -Weights.wYdelta * dPixelCenRow;
		pdRowBegin[iNextCamFovPos+4]	= -Weights.wYdelta * dPixelCenCol;
		pdRowBegin[iNextCamFovPos+5]	= -Weights.wYdelta;

		if(_bProjectiveTrans)
		{
			pdRowBegin[iFOVPos+9]		= Weights.wYdelta * dPixelCenRow* dPixelCenRow;
			pdRowBegin[iFOVPos+10]		= Weights.wYdelta * dPixelCenRow* dPixelCenCol;
			pdRowBegin[iFOVPos+11]		= Weights.wYdelta * dPixelCenCol * dPixelCenCol;
			pdRowBegin[iNextCamFovPos+9]	= -Weights.wYdelta * dPixelCenRow* dPixelCenRow;
			pdRowBegin[iNextCamFovPos+10]	= -Weights.wYdelta * dPixelCenRow* dPixelCenCol;
			pdRowBegin[iNextCamFovPos+11]	= -Weights.wYdelta * dPixelCenCol * dPixelCenCol;
		}
		_dVectorB[_iCurrentRow] = Weights.wYdelta * (dFovCalCenY - dNextCamFovCalCenY);
		pdRowBegin += _iMatrixWidth;
		_iCurrentRow++;
	}

	//* 9 distance between cameras in X
	if(iNextCamFovPos > 0)
	{
		pdRowBegin[iFOVPos+0]		= Weights.wXdelta * dPixelCenRow;
		pdRowBegin[iFOVPos+1]		= Weights.wXdelta * dPixelCenCol;
		pdRowBegin[iFOVPos+2]		= Weights.wXdelta;
		pdRowBegin[iNextCamFovPos+0]	= -Weights.wXdelta * dPixelCenRow;
		pdRowBegin[iNextCamFovPos+1]	= -Weights.wXdelta * dPixelCenCol;
		pdRowBegin[iNextCamFovPos+2]	= -Weights.wXdelta;

		if(_bProjectiveTrans)
		{
			pdRowBegin[iFOVPos+6]		= Weights.wXdelta * dPixelCenRow* dPixelCenRow;
			pdRowBegin[iFOVPos+7]		= Weights.wXdelta * dPixelCenRow* dPixelCenCol;
			pdRowBegin[iFOVPos+8]		= Weights.wXdelta * dPixelCenCol * dPixelCenCol;
			pdRowBegin[iNextCamFovPos+6]	= -Weights.wXdelta * dPixelCenRow* dPixelCenRow;
			pdRowBegin[iNextCamFovPos+7]	= -Weights.wXdelta * dPixelCenRow* dPixelCenCol;
			pdRowBegin[iNextCamFovPos+8]	= -Weights.wXdelta * dPixelCenCol * dPixelCenCol;
		}
		_dVectorB[_iCurrentRow] = Weights.wXdelta * (dFovCalCenX - dNextCamFovCalCenX);
		pdRowBegin += _iMatrixWidth;
		_iCurrentRow++;
	}

	//* 10 Position of the FOV in X 
	pdRowBegin[iFOVPos+2] = Weights.wXIndexwt;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = Weights.wXIndexwt * transFov.GetItem(2);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	// For projective transform
	if(_bProjectiveTrans)
	{	
		//* 11 M6 = M10
		pdRowBegin[iFOVPos+6] = Weights.wPMEq;
		pdRowBegin[iFOVPos+10] = -Weights.wPMEq;
					
		//* 12 M7 = M11
		pdRowBegin[iFOVPos+7] = Weights.wPMEq;
		pdRowBegin[iFOVPos+11]= -Weights.wPMEq;
					
		//* 13 M8 = 0
		pdRowBegin[iFOVPos+8] = Weights.wPM89;

		//* 14 M9 = 0;
		pdRowBegin[iFOVPos+9] = Weights.wPM89;

					
		if(iNextCamFovPos > 0 )
		{	
			//* 15 M10 = Next FOV M10
			pdRowBegin[iFOVPos+10] = Weights.wPMNext;
			pdRowBegin[iNextCamFovPos+10] = -Weights.wPMNext;
						
			//* 16 M11 = Next FOV M11
			pdRowBegin[iFOVPos+11] = Weights.wPMNext;
			pdRowBegin[iNextCamFovPos+11] = -Weights.wPMNext;
		}
	}

	return(true);
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