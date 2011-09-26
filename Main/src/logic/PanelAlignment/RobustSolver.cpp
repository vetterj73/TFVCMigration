#include "RobustSolver.h"
#include "EquationWeights.h"
#include "lsqrpoly.h"
#include "Logger.h"

#pragma region Operateors
bool operator<(const FovIndex& a, const FovIndex& b)
{
	if(a.IlluminationIndex < b.IlluminationIndex)
		return (true);
	else if(a.IlluminationIndex > b.IlluminationIndex)
		return (false);
	else
	{
		if(a.TriggerIndex < b.TriggerIndex)
			return(true);
		else if(a.TriggerIndex > b.TriggerIndex)
			return(false);
		else
		{ 
			if(a.CameraIndex < b.CameraIndex)
				return(true);
			else if(a.CameraIndex > b.CameraIndex)
				return(false);
		}
	}

	return(false);
}

bool operator>(const FovIndex& a, const FovIndex& b)
{
	if(a.IlluminationIndex > b.IlluminationIndex)
		return (true);
	else if(a.IlluminationIndex < b.IlluminationIndex)
		return(false);
	else
	{
		if(a.TriggerIndex > b.TriggerIndex)
			return(true);
		else if(a.TriggerIndex < b.TriggerIndex)
			return(false);
		else
		{
			if(a.CameraIndex > b.CameraIndex)
				return(true);
			else if(a.CameraIndex < b.CameraIndex)
				return(false);
		}
	}
			
	return(false);
}
#pragma endregion

#pragma region constructor

RobustSolver::RobustSolver(		
	map<FovIndex, unsigned int>* pFovOrderMap, 
	unsigned int iMaxNumCorrelation, 
	bool bProjectiveTrans)
{
	_pFovOrderMap = pFovOrderMap;
	_bProjectiveTrans = bProjectiveTrans;	
	
	_iNumFovs = (unsigned int)pFovOrderMap->size();
	
	if(_bProjectiveTrans)	// For projective transform
	{		
		_iNumParamsPerFov = 12;			// parameters per Fov
		_iNumCalibConstrains = 17;		// calibration related constraints
	}
	else		// For Affine transform
	{
		_iNumParamsPerFov = 6;			// parameters per Fov
		_iNumCalibConstrains = 11;		// calibration related constraints
	}

	_iMatrixWidth = _iNumFovs * _iNumParamsPerFov;
	_iMatrixHeight = _iNumFovs * _iNumCalibConstrains + 2*iMaxNumCorrelation;

	_iMatrixSize = _iMatrixWidth * _iMatrixHeight;

	_dMatrixA = new double[_iMatrixSize];
	_dMatrixACopy = new double[_iMatrixSize];
	
	_dVectorB = new double[_iMatrixHeight];
	_dVectorBCopy = new double[_iMatrixHeight];

	_dVectorX = new double[_iMatrixWidth];

	ZeroTheSystem();
}

RobustSolver::~RobustSolver(void)
{
	delete [] _dMatrixA;
	delete [] _dMatrixACopy;
	delete [] _dVectorB;
	delete [] _dVectorBCopy;
	delete [] _dVectorX;
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

#pragma endregion

#pragma region Add equations
// Add Constraints for one image
bool RobustSolver::AddCalibationConstraints(MosaicLayer* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex, bool bUseFiducials)
{
	// Validation check
	if(iCamIndex>=pMosaic->GetNumberOfCameras() || iTrigIndex>=pMosaic->GetNumberOfTriggers())
		return(false);

	// Fov transform parameter begin position in column
	// Fov's nominal center
	FovIndex index(pMosaic->Index(), iTrigIndex, iCamIndex); 
	int iFOVPos = (*_pFovOrderMap)[index] *_iNumParamsPerFov;
	ImgTransform transFov = pMosaic->GetImage(iCamIndex, iTrigIndex)->GetNominalTransform();
	unsigned int iCols = pMosaic->GetImage(iCamIndex, iTrigIndex)->Columns();
	unsigned int iRows = pMosaic->GetImage(iCamIndex, iTrigIndex)->Rows();
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
	if(index.CameraIndex < pMosaic->GetNumberOfCameras())
	{
		iNextCamFovPos = (*_pFovOrderMap)[index] * _iNumParamsPerFov;
		transNextCamFov = pMosaic->GetImage(index.CameraIndex, index.TriggerIndex)->GetNominalTransform();
		transNextCamFov.Map(dPixelCenRow, dPixelCenCol, &dNextCamFovCalCenX, &dNextCamFovCalCenY);
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
		transNextTrigFov = pMosaic->GetImage(index.CameraIndex, index.TriggerIndex)->GetNominalTransform();
		transNextTrigFov.Map(dPixelCenRow, dPixelCenCol, &dNextTrigFovCalCenX, &dNextTrigFovCalCenY);
	}*/

	double* pdRowBegin = _dMatrixA + _iCurrentRow*_iMatrixWidth;

	/*
	Affine transform
		[	M0	M1	M2	]	
		[	M3	M4	M5	]
	Projective transform
		[	M0	M1	M2	M6	M7	M8	]	
		[	M3	M4	M5	M9	M10	M11	]
	*/
	//* 1 rotation consist (M1+M3=0)
	pdRowBegin[iFOVPos+1] = Weights.wRxy;
	pdRowBegin[iFOVPos+3] = Weights.wRxy;
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 2  Square pixel (M0-M4=0)
	pdRowBegin[iFOVPos+0] = Weights.wMxy;
	pdRowBegin[iFOVPos+4] = -Weights.wMxy;
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 3 rotation match (M1 = cal[1]) 
	pdRowBegin[iFOVPos+1] = Weights.wRcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = Weights.wRcal * transFov.GetItem(1);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 4 rotate match (M3 = cal[3])
	pdRowBegin[iFOVPos+3] = Weights.wRcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = Weights.wRcal * transFov.GetItem(3);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 5 pixel size match in Y (M0 = cal[0]) 
	pdRowBegin[iFOVPos+0] = Weights.wMcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = Weights.wMcal * transFov.GetItem(0);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 6 pixel size match in X (M4 = cal[4])
	pdRowBegin[iFOVPos+4] = Weights.wMcal;	// b VALUE IS NON_ZERO!!
	_dVectorB[_iCurrentRow] = Weights.wMcal * transFov.GetItem(4);
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 7 (M1-next camera M1) match Calib
	if(iNextCamFovPos > 0) // Next camera exists
	{
		pdRowBegin[iFOVPos+1] = Weights.wYRdelta;
		pdRowBegin[iNextCamFovPos+1] = - Weights.wYRdelta;
		_dVectorB[_iCurrentRow] = Weights.wYRdelta * (transFov.GetItem(1) - transNextCamFov.GetItem(1));
		pdRowBegin += _iMatrixWidth;
		_iCurrentRow++;
	}

	//* 8 fov center Y pos match Cal
	// Use FOV center instread of (0,0) is to add more constraint for projective transform
	double dTempWeight = Weights.wYcent;
	// If fiducial information is not used, one FOV will have high weight
	if(!bUseFiducials && pMosaic->Index()==0 && iTrigIndex == 1 && iCamIndex == 1)
		dTempWeight = Weights.wYcentNoFid;
	pdRowBegin[iFOVPos+3] = dTempWeight * dPixelCenRow;
	pdRowBegin[iFOVPos+4] = dTempWeight * dPixelCenCol;
	pdRowBegin[iFOVPos+5] = dTempWeight;
	if(_bProjectiveTrans)
	{
		pdRowBegin[iFOVPos+9] = dTempWeight * dPixelCenRow * dPixelCenRow;
		pdRowBegin[iFOVPos+10] = dTempWeight * dPixelCenRow * dPixelCenCol;
		pdRowBegin[iFOVPos+11] = dTempWeight * dPixelCenCol * dPixelCenCol;
	}
	_dVectorB[_iCurrentRow] = dTempWeight  * dFovCalCenY;
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 9 Position of the FOV in X match Cal
	// Use FOV center instread of (0,0) is to add more constraint for projective transform
	dTempWeight = Weights.wXcent;
	// If fiducial information is not used, one FOV will have high weight
	if(!bUseFiducials && pMosaic->Index()==0 && iTrigIndex == 1 && iCamIndex == 1)
		dTempWeight = Weights.wXcentNoFid;
	pdRowBegin[iFOVPos+0] = dTempWeight * dPixelCenRow;
	pdRowBegin[iFOVPos+1] = dTempWeight * dPixelCenCol;
	pdRowBegin[iFOVPos+2] = dTempWeight;
	if(_bProjectiveTrans)
	{
		pdRowBegin[iFOVPos+6] = dTempWeight * dPixelCenRow * dPixelCenRow;
		pdRowBegin[iFOVPos+7] = dTempWeight * dPixelCenRow * dPixelCenCol;
		pdRowBegin[iFOVPos+8] = dTempWeight * dPixelCenCol * dPixelCenCol;
	}
	_dVectorB[_iCurrentRow] = dTempWeight  * dFovCalCenX;
	pdRowBegin += _iMatrixWidth;
	_iCurrentRow++;

	//* 10 distance between cameras in Y
	if(iNextCamFovPos > 0) // Next camera exists
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

	//* 11 distance between cameras in X
	if(iNextCamFovPos > 0) // Next camera exists
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

	// For projective transform
	if(_bProjectiveTrans)
	{	
		//* 12 M6 = M10
		pdRowBegin[iFOVPos+6] = Weights.wPMEq;
		pdRowBegin[iFOVPos+10] = -Weights.wPMEq;
		pdRowBegin += _iMatrixWidth;
		_iCurrentRow++;
					
		//* 13 M7 = M11
		pdRowBegin[iFOVPos+7] = Weights.wPMEq;
		pdRowBegin[iFOVPos+11]= -Weights.wPMEq;
		pdRowBegin += _iMatrixWidth;
		_iCurrentRow++;
					
		//* 14 M8 = 0
		pdRowBegin[iFOVPos+8] = Weights.wPM89;
		pdRowBegin += _iMatrixWidth;
		_iCurrentRow++;

		//* 15 M9 = 0;
		pdRowBegin[iFOVPos+9] = Weights.wPM89;
		pdRowBegin += _iMatrixWidth;
		_iCurrentRow++;
							
		if(iNextCamFovPos > 0 ) // Next camera exists
		{	
			//* 16 M10 = Next camera M10
			pdRowBegin[iFOVPos+10] = Weights.wPMNext;
			pdRowBegin[iNextCamFovPos+10] = -Weights.wPMNext;
			pdRowBegin += _iMatrixWidth;
			_iCurrentRow++;
						
			//* 17 M11 = Next camera M11
			pdRowBegin[iFOVPos+11] = Weights.wPMNext;
			pdRowBegin[iNextCamFovPos+11] = -Weights.wPMNext;
			pdRowBegin += _iMatrixWidth;
			_iCurrentRow++;
		}
	}

	return(true);
}

// Add results for one Fov and Fov overlap
bool RobustSolver::AddFovFovOvelapResults(FovFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);

	// First Fov's information
	unsigned int iMosicIndexA = pOverlap->GetFirstMosaicImage()->Index();
	unsigned int iTrigIndexA = pOverlap->GetFirstTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetFirstCameraIndex();
	FovIndex index1(iMosicIndexA, iTrigIndexA, iCamIndexA); 
	unsigned int iFOVPosA = (*_pFovOrderMap)[index1] *_iNumParamsPerFov;

	// Second Fov's information
	unsigned int iMosicIndexB = pOverlap->GetSecondMosaicImage()->Index();
	unsigned int iTrigIndexB = pOverlap->GetSecondTriggerIndex();
	unsigned int iCamIndexB = pOverlap->GetSecondCameraIndex();
	FovIndex index2(iMosicIndexB, iTrigIndexB, iCamIndexB); 
	unsigned int iFOVPosB = (*_pFovOrderMap)[index2] *_iNumParamsPerFov;

	double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;

	list<CorrelationPair>* pPairList = pOverlap->GetFinePairListPtr();
	for(list<CorrelationPair>::iterator i= pPairList->begin(); i!=pPairList->end(); i++)
	{
		// Skip any fine that is not processed or not good
		if(!i->IsProcessed() || !i->IsGoodForSolver())
			continue;

		// validation check for correlation pair
		CorrelationResult result;
		bool bFlag = i->GetCorrelationResult(&result);
		if(!bFlag) 
			continue;

		double w = Weights.CalWeight(&(*i));
		if(w <= 0) 
			continue;

		// Get Centers of ROIs
		double rowImgA = (i->GetFirstRoi().FirstRow + i->GetFirstRoi().LastRow)/ 2.0;
		double colImgA = (i->GetFirstRoi().FirstColumn + i->GetFirstRoi().LastColumn)/ 2.0;

		double rowImgB = (i->GetSecondRoi().FirstRow + i->GetSecondRoi().LastRow)/ 2.0;
		double colImgB = (i->GetSecondRoi().FirstColumn + i->GetSecondRoi().LastColumn)/ 2.0;

		// Get offset
		double offsetRows = result.RowOffset;
		double offsetCols = result.ColOffset;

		// Add a equataion for X
		pdRow[iFOVPosA + 0] = (rowImgA-offsetRows) * w;
		pdRow[iFOVPosA + 1] = (colImgA-offsetCols) * w;
		pdRow[iFOVPosA + 2] = w;
		pdRow[iFOVPosB + 0] = -rowImgB * w;
		pdRow[iFOVPosB + 1] = -colImgB * w;
		pdRow[iFOVPosB + 2] = -w;

		// For projective transform
		if(_bProjectiveTrans)
		{
			pdRow[iFOVPosA + 6] = (rowImgA-offsetRows) * (rowImgA-offsetRows) * w;
			pdRow[iFOVPosA + 7] = (rowImgA-offsetRows) * (colImgA-offsetCols) * w;
			pdRow[iFOVPosA + 8] = (colImgA-offsetCols) * (colImgA-offsetCols) * w;
			pdRow[iFOVPosB + 6] = -rowImgB * rowImgB * w;
			pdRow[iFOVPosB + 7] = -rowImgB * colImgB * w;
			pdRow[iFOVPosB + 8] = -colImgB * colImgB * w;
		}
		pdRow += _iMatrixWidth;
		_iCurrentRow++;

		// Add a equataion for Y
		pdRow[iFOVPosA + 3] = (rowImgA-offsetRows) * w;
		pdRow[iFOVPosA + 4] = (colImgA-offsetCols) * w;
		pdRow[iFOVPosA + 5] = w;
		pdRow[iFOVPosB + 3] = -rowImgB * w;
		pdRow[iFOVPosB + 4] = -colImgB * w;
		pdRow[iFOVPosB + 5] = -w;

		// For projective transform
		if(_bProjectiveTrans)
		{
			pdRow[iFOVPosA + 9] = (rowImgA-offsetRows) * (rowImgA-offsetRows) * w;
			pdRow[iFOVPosA +10] = (rowImgA-offsetRows) * (colImgA-offsetCols) * w;
			pdRow[iFOVPosA +11] = (colImgA-offsetCols) * (colImgA-offsetCols) * w;
			pdRow[iFOVPosB + 9] = -rowImgB * rowImgB * w;
			pdRow[iFOVPosB +10] = -rowImgB * colImgB * w;
			pdRow[iFOVPosB +11] = -colImgB * colImgB * w;
		}
		pdRow += _iMatrixWidth;
		_iCurrentRow++;
	}

	return(true);
}

// Add results for one Cad and Fov overlap
bool RobustSolver::AddCadFovOvelapResults(CadFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);

	// Fov's information
	unsigned int iMosicIndex= pOverlap->GetMosaicImage()->Index();
	unsigned int iTrigIndex = pOverlap->GetTriggerIndex();
	unsigned int iCamIndex = pOverlap->GetCameraIndex();
	FovIndex index(iMosicIndex, iTrigIndex, iCamIndex); 
	unsigned int iFOVPosA = (*_pFovOrderMap)[index] *_iNumParamsPerFov;

	double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;

	CorrelationPair* pPair = pOverlap->GetCoarsePair();

	// validation check for correlation pair
	CorrelationResult result;
	bool bFlag = pPair->GetCorrelationResult(&result);
	if(!bFlag) 
		return(false);

	double w = Weights.CalWeight(pPair);
	if(w <= 0) 
		return(false);

	// Get Centers of ROIs
	double rowImgA = (pPair->GetFirstRoi().FirstRow + pPair->GetFirstRoi().LastRow)/ 2.0;
	double colImgA = (pPair->GetFirstRoi().FirstColumn + pPair->GetFirstRoi().LastColumn)/ 2.0;

	double rowImgB = (pPair->GetSecondRoi().FirstRow + pPair->GetSecondRoi().LastRow)/ 2.0;
	double colImgB = (pPair->GetSecondRoi().FirstColumn + pPair->GetSecondRoi().LastColumn)/ 2.0;
	double dCadCenX, dCadCenY;
	pOverlap->GetCadImage()->ImageToWorld(rowImgB, colImgB, &dCadCenX, &dCadCenY);

	// Get offset
	double offsetRows = result.RowOffset;
	double offsetCols = result.ColOffset;

	// Add a equataion for X
	pdRow[iFOVPosA + 0] = (rowImgA-offsetRows) * w;
	pdRow[iFOVPosA + 1] = (colImgA-offsetCols) * w;
	pdRow[iFOVPosA + 2] = w;

	// For projective transform
	if(_bProjectiveTrans)
	{
		pdRow[iFOVPosA + 6] = (rowImgA-offsetRows) * (rowImgA-offsetRows) * w;
		pdRow[iFOVPosA + 7] = (rowImgA-offsetRows) * (colImgA-offsetCols) * w;
		pdRow[iFOVPosA + 8] = (colImgA-offsetCols) * (colImgA-offsetCols) * w;
	}
	_dVectorB[_iCurrentRow] = w*dCadCenX;
	pdRow += _iMatrixWidth;
	_iCurrentRow++;

	// Add a equataion for Y
	pdRow[iFOVPosA + 3] = (rowImgA-offsetRows) * w;
	pdRow[iFOVPosA + 4] = (colImgA-offsetCols) * w;
	pdRow[iFOVPosA + 5] = w;

	// For projective transform
	if(_bProjectiveTrans)
	{
		pdRow[iFOVPosA + 9] = (rowImgA-offsetRows) * (rowImgA-offsetRows) * w;
		pdRow[iFOVPosA +10] = (rowImgA-offsetRows) * (colImgA-offsetCols) * w;
		pdRow[iFOVPosA +11] = (colImgA-offsetCols) * (colImgA-offsetCols) * w;
	}
	_dVectorB[_iCurrentRow] = w*dCadCenY;
	pdRow += _iMatrixWidth;
	_iCurrentRow++;
	
	return(true);
}

// Add results for one Fiducial and Fov overlap
bool RobustSolver::AddFidFovOvelapResults(FidFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);

	// Fov's information
	unsigned int iMosicIndex= pOverlap->GetMosaicImage()->Index();
	unsigned int iTrigIndex = pOverlap->GetTriggerIndex();
	unsigned int iCamIndex = pOverlap->GetCameraIndex();
	FovIndex index(iMosicIndex, iTrigIndex, iCamIndex); 
	unsigned int iFOVPosA = (*_pFovOrderMap)[index] *_iNumParamsPerFov;

	double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;

	CorrelationPair* pPair = pOverlap->GetCoarsePair();

	// Validation check for correlation pair
	CorrelationResult result;
	bool bFlag = pPair->GetCorrelationResult(&result);
	if(!bFlag) 
		return(false);

	double w = Weights.CalWeight(pPair);
	if(w <= 0) 
		return(false);

	// Get Centers of ROIs (fiducial image is always the second one)
	double rowImgA = (pPair->GetFirstRoi().FirstRow + pPair->GetFirstRoi().LastRow)/ 2.0;
	double colImgA = (pPair->GetFirstRoi().FirstColumn + pPair->GetFirstRoi().LastColumn)/ 2.0;

	double rowImgB = (pPair->GetSecondRoi().FirstRow + pPair->GetSecondRoi().LastRow)/ 2.0;
	double colImgB = (pPair->GetSecondRoi().FirstColumn + pPair->GetSecondRoi().LastColumn)/ 2.0;
	double dFidRoiCenX, dFidRoiCenY; // ROI center of fiducial image (not fiducail center or image center) in world space
	pOverlap->GetFidImage()->ImageToWorld(rowImgB, colImgB, &dFidRoiCenX, &dFidRoiCenY);

	// Get offset
	double offsetRows = result.RowOffset;
	double offsetCols = result.ColOffset;

	// Add a equataion for X
	pdRow[iFOVPosA + 0] = (rowImgA-offsetRows) * w;
	pdRow[iFOVPosA + 1] = (colImgA-offsetCols) * w;
	pdRow[iFOVPosA + 2] = w;

	// For projective transform
	if(_bProjectiveTrans)
	{
		pdRow[iFOVPosA + 6] = (rowImgA-offsetRows) * (rowImgA-offsetRows) * w;
		pdRow[iFOVPosA + 7] = (rowImgA-offsetRows) * (colImgA-offsetCols) * w;
		pdRow[iFOVPosA + 8] = (colImgA-offsetCols) * (colImgA-offsetCols) * w;
	}
	_dVectorB[_iCurrentRow] = w*dFidRoiCenX;
	pdRow += _iMatrixWidth;
	_iCurrentRow++;

	// Add a equataion for Y
	pdRow[iFOVPosA + 3] = (rowImgA-offsetRows) * w;
	pdRow[iFOVPosA + 4] = (colImgA-offsetCols) * w;
	pdRow[iFOVPosA + 5] = w;

	// For projective transform
	if(_bProjectiveTrans)
	{
		pdRow[iFOVPosA + 9] = (rowImgA-offsetRows) * (rowImgA-offsetRows) * w;
		pdRow[iFOVPosA +10] = (rowImgA-offsetRows) * (colImgA-offsetCols) * w;
		pdRow[iFOVPosA +11] = (colImgA-offsetCols) * (colImgA-offsetCols) * w;
	}
	_dVectorB[_iCurrentRow] = w*dFidRoiCenY;
	pdRow += _iMatrixWidth;
	_iCurrentRow++;
	
	return(true);
}
#pragma endregion

#pragma region Solver and transform
struct LeftIndex
{
	unsigned int iLeft;
	unsigned int iIndex;
	unsigned int iLength;
};

bool operator<(const LeftIndex& a, const LeftIndex& b)
{
	return(a.iLeft < b.iLeft);
};

// Reorder the matrix A and vector B, and transpose Matrix A for banded solver
// bRemoveEmptyRows, flag for removing empty equations from Matrix A
// piCounts: output, # rows in block i for banded solver
// piEmptyRows: output, number of empty equations in Matrix A
unsigned int RobustSolver::ReorderAndTranspose(bool bRemoveEmptyRows, int* piCounts, unsigned int* piEmptyRows)
{
	//Get map for reorder
	list<LeftIndex> leftIndexList;
	int iStart, iEnd;
	bool bFlag;
	int iMaxIndex;
	unsigned int iMaxLength = 0;	// Max unempty length in a row 
	*piEmptyRows = 0;				// Number of empty rows
	for(unsigned int row(0); row<_iMatrixHeight; ++row) // for each row
	{
		bFlag = false;  // False if the whole row is zero
		for(unsigned int col(0); col<_iMatrixWidth; ++col)
		{
			if(_dMatrixA[row*_iMatrixWidth+col]!=0)
			{
				if(!bFlag)
				{
					iStart = col; //record the first column
					bFlag = true;
				}

				iEnd = col; // record the last column
			}
		}

		LeftIndex node;				
		node.iIndex = row;
		if(bFlag) // Not a empty row
		{
			node.iLeft = iStart;
			node.iLength = iEnd-iStart+1;
		}
		else
		{	// Empty Rows
			node.iLeft = _iMatrixWidth+1; // Empty rows need put on the last 
			node.iLength = 0;

			(*piEmptyRows)++;
		}
		
		// Update iMaxLength
		if(node.iLength > iMaxLength)
		{
			iMaxLength = node.iLength;
			iMaxIndex = row;
		}
		
		leftIndexList.push_back(node);	
	}
	// Sort node
	leftIndexList.sort();

	// Clear piCount
	for(unsigned int j = 0; j<_iMatrixWidth; j++)
		piCounts[j] = 0;

	// Reorder and transpose
	double* workspace = new double[_iMatrixSize];
	double* dCopyB = new double[_iMatrixHeight];
	unsigned int iDestRow = 0;
	list<LeftIndex>::const_iterator i;
	for(i=leftIndexList.begin(); i!=leftIndexList.end(); i++)
	{
		for(unsigned int col(0); col<_iMatrixWidth; ++col)
			workspace[col*_iMatrixHeight+iDestRow] = _dMatrixA[i->iIndex*_iMatrixWidth+col];

		dCopyB[iDestRow] = _dVectorB[i->iIndex];

		// Skip the empty rows
		if(bRemoveEmptyRows && (i->iLength == 0)) continue;

		// Count Rows of each block 
		if(i->iLeft < _iMatrixWidth - iMaxLength)
			piCounts[i->iLeft]++; // All blocks except the last one
		else
			piCounts[_iMatrixWidth - iMaxLength]++; // The last block
		
		iDestRow++;
	}
	
	unsigned int temp=0;
	for(unsigned int k=0; k<_iMatrixWidth-iMaxLength+1; k++)
		temp += piCounts[k];

	if(bRemoveEmptyRows)
	{
		// Skip the empty rows in input matrix (columns in workspace)
		for(unsigned int k=0; k<_iMatrixWidth; k++)
			::memcpy(_dMatrixA+k*(_iMatrixHeight-*piEmptyRows), workspace+k*_iMatrixHeight, (_iMatrixHeight-*piEmptyRows)*sizeof(double));

	}
	else	// include empty rows
		::memcpy(_dMatrixA, workspace, _iMatrixSize*sizeof(double));

	for(unsigned int k=0; k<_iMatrixHeight; k++)
		_dVectorB[k] = dCopyB[k];

 /*/ for debug
	// Save transposed Matrix A 
	ofstream of("C:\\Temp\\MatrixA_t.csv");
	of << std::scientific;

	int ilines = _iMatrixHeight;
	if(bRemoveEmptyRows)
		ilines = _iMatrixHeight-*piEmptyRows;

	for(unsigned int j=0; j<_iMatrixWidth; j++)
	{
		for(unsigned int k=0; k<ilines; k++)
		{ 
	
			of << _dMatrixA[j*ilines+k];
			if(j != ilines-1)
				of <<",";
		}
		of << std::endl;
	}

	of.close();

	// Save Matrix A
	of.open("C:\\Temp\\MatrixA.csv");		
	for(unsigned int k=0; k<ilines; k++)
	{ 
		for(unsigned int j=0; j<_iMatrixWidth; j++)
		{
			of << _dMatrixA[j*ilines+k];
			if(j != _iMatrixWidth-1)
				of <<",";
		}
		of << std::endl;
	}
	of.close();

	// Save Matrix B
	of.open("C:\\Temp\\VectorB.csv");
	for(unsigned int k=0; k<ilines; k++)
	{ 
		of << _dVectorB[k] << std::endl;
	}
	of.close();

	// Save blocklength
	of.open("C:\\Temp\\BlockLength.csv");
	of << _iMatrixWidth << std::endl;
	of << ilines <<std::endl;
	of << iMaxLength <<std::endl;
	for(unsigned int k=0; k<_iMatrixWidth; k++)
	{ 
		of <<  piCounts[k] << std::endl;
	}
	of.close();
//*/

	delete [] workspace;
	delete [] dCopyB;

	return(iMaxLength);
}

// Robust regression by Huber's "Algorithm H"
// Banded version
void RobustSolver::SolveXAlgHB()
{

	::memcpy(_dMatrixACopy, _dMatrixA, _iMatrixSize*sizeof(double));
	::memcpy(_dVectorBCopy, _dVectorB, _iMatrixHeight*sizeof(double)); 

	// we built A in row order
	// the qr factorization method requires column order
	bool bRemoveEmptyRows = true;
	int* mb = new int[_iMatrixWidth];
	unsigned int iEmptyRows;
	unsigned int bw = ReorderAndTranspose(bRemoveEmptyRows, mb, &iEmptyRows);

	double*	resid = new double[_iMatrixHeight];
	double scaleparm = 0;
	double cond = 0;

	LOG.FireLogEntry(LogTypeSystem, "RobustSolver::SolveXAlgHB():BEGIN ALG_HB");

	int algHRetVal = 
		alg_hb(                // Robust regression by Huber's "Algorithm H"/ Banded version.
			// Inputs //			// See qrdb() in qrband.c for more information.

			bw,						// Width (bandwidth) of each block 
			_iMatrixWidth-bw+1,		// # of blocks 
			mb,						// mb[i] = # rows in block i 
			_dMatrixA,				// System matrix
			_dVectorB,			// Constant vector (not overwritten); must not coincide with x[] or resid[]. 

		   // Outputs //

			_dVectorX,            // Coefficient vector (overwritten with solution) 
			resid,				// Residuals b - A*x (pass null if not wanted).  If non-null, must not coincide with x[] or b[].
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



	if( algHRetVal<0 )
		LOG.FireLogEntry(LogTypeSystem, "RobustSolver::SolveXAlgHB():alg_hb returned value of %d", algHRetVal);

	LOG.FireLogEntry(LogTypeSystem, "RobustSolver::SolveXAlgHB():FINISHED ALG_HB");
	LOG.FireLogEntry(LogTypeSystem, "RobustSolver::SolveXAlgHB():alg_hb Bandwidth = %d", bw);
	LOG.FireLogEntry(LogTypeSystem, "RobustSolver::SolveXAlgHB():alg_hb nIterations=%d, scaleparm=%f, cond=%f", algHRetVal, scaleparm, cond);

	delete [] mb;
	delete [] resid;
}

// populates object referred to by the function argument t with 
// the terms that define the transformation
ImgTransform RobustSolver::GetResultTransform(
	unsigned int iLlluminationIndex,
	unsigned int iTriggerIndex,
	unsigned int iCameraIndex) const 
{
	// Fov transform parameter begin position in column
	FovIndex fovIndex(iLlluminationIndex, iTriggerIndex, iCameraIndex); 
	unsigned int index = (*_pFovOrderMap)[fovIndex] *_iNumParamsPerFov;

	double t[3][3];

	if(!_bProjectiveTrans)
	{

		t[0][0] = _dVectorX[ index + 0 ];
		t[0][1] = _dVectorX[ index + 1 ];
		t[0][2] = _dVectorX[ index + 2 ];
			   
		t[1][0] = _dVectorX[ index + 3 ];
		t[1][1] = _dVectorX[ index + 4 ];
		t[1][2] = _dVectorX[ index + 5 ];
			   
		t[2][0] = 0;
		t[2][1] = 0;
		t[2][2] = 1;
	}
	else
	{
		// Create matching projetive transform
		MatchProjeciveTransform(&_dVectorX[index], t);
	}

	ImgTransform trans(t);

	return(trans);
}

// From 12 calcualted parameters to match a projecive transform
bool RobustSolver::MatchProjeciveTransform(const double pPara[12], double dTrans[3][3]) const
{
	int iNumX = 10;
	int iNumY = 10;
	int iStepX = 1944/iNumX;  // Rows
	int iStepY = 2592/iNumY;  // Columes
	double* pRow = new double[iNumX*iNumY];
	double* pCol = new double[iNumX*iNumY];
	double* pX	 = new double[iNumX*iNumY];
	double* pY	 = new double[iNumX*iNumY];

	// Create match pairs for projective transform
	int iIndex = 0;
	for(int ky=0; ky<iNumY; ky++)
	{
		for(int kx=0; kx<iNumX; kx++)
		{
			pRow[iIndex] = kx*iStepX;
			pCol[iIndex] = ky*iStepY;

			pX[iIndex] = pRow[iIndex]*pPara[0] + pCol[iIndex]*pPara[1] + pPara[2] +
				pRow[iIndex]*pRow[iIndex]*pPara[6] +
				pRow[iIndex]*pCol[iIndex]*pPara[7] +
				pCol[iIndex]*pCol[iIndex]*pPara[8];

			pY[iIndex] = pRow[iIndex]*pPara[3] + pCol[iIndex]*pPara[4] + pPara[5] +
				pRow[iIndex]*pRow[iIndex]*pPara[9] +
				pRow[iIndex]*pCol[iIndex]*pPara[10] +
				pCol[iIndex]*pCol[iIndex]*pPara[11];

			iIndex++;
		}
	}

	// Create matching projective transform
	int m = iNumX*iNumY;
	complexd* p = new complexd[m];
	complexd* q = new complexd[m];
	complexd* resid = new complexd[m];
	//complexd p[25], q[25], resid[25];
	double RMS, dRcond;
	
	for(int j=0; j<m; j++)
	{
		p[j].r = pRow[j];
		p[j].i = pCol[j];
		q[j].r = pX[j];
		q[j].i = pY[j];
	}

	int iFlag = lsqrproj(m, p, q, 1, dTrans, &RMS, resid, &dRcond);
	//LOG.FireLogEntry(LogTypeSystem, "RobustSolver::MatchProjeciveTransform():RMS = %f", RMS/(1.6e-5));

	if(iFlag != 0)
	{
		//LOG.FireLogEntry(LogTypeSystem, "RobustSolver::MatchProjeciveTransform():Failed to create matching projective transform");
	}

	delete [] p;
	delete [] q;
	delete [] resid;

	delete [] pRow;
	delete [] pCol;
	delete [] pX;
	delete [] pY;

	if(iFlag != 0)
		return(false);
	else
		return(true);
}

#pragma endregion

#pragma region Debug
// Debug
// Vector X output
void RobustSolver::OutputVectorXCSV(string filename) const
{
	ofstream of(filename.c_str());

	of << std::scientific;

	string line;

	for(map<FovIndex, unsigned int>::iterator k=_pFovOrderMap->begin(); k!=_pFovOrderMap->end(); k++)
	{
		of << "I_" << k->first.IlluminationIndex 
			<< "T_" << k->first.TriggerIndex 
			<< "C_" << k->first.CameraIndex
			<< ",";

		for(unsigned int j=0; j<_iNumParamsPerFov; ++j)
		{
			if( j!=0 )
				of << ",";

			double d = _dVectorX[k->second*_iNumParamsPerFov + j];

			of << d;
		}

		of <<  std::endl;
	}

	of.close();
}
#pragma endregion
