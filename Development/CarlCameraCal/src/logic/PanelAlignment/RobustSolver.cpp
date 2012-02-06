#include "RobustSolver.h"
#include "EquationWeights.h"
#include "lsqrpoly.h"
#include "Logger.h"

#include "rtypes.h"
#include "dot2d.h"
extern "C" {
#include "ucxform.h"
#include "warpxy.h"
}


// camera model distortion array horizontal direction is index 0 == CAD direction Y
// index 1 == CAD direction X
#define CAL_ARRAY_X			1
#define CAL_ARRAY_Y			0


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
	map<FovIndex, unsigned int>* pFovOrderMap)
{
	_pFovOrderMap = pFovOrderMap;
	_bSaveMatrixCSV=false;
	_bVerboseLogging = false;
	_iNumFovs = (unsigned int)pFovOrderMap->size();
	
}
RobustSolverFOV::RobustSolverFOV(		
	map<FovIndex, unsigned int>* pFovOrderMap, 
	unsigned int iMaxNumCorrelations, 
	bool bProjectiveTrans): 	RobustSolver( pFovOrderMap)
{	
	_bProjectiveTrans = bProjectiveTrans;
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
	_iMatrixHeight = _iNumFovs * _iNumCalibConstrains + 2*iMaxNumCorrelations;

	_iMatrixSize = _iMatrixWidth * _iMatrixHeight;

	_dMatrixA = new double[_iMatrixSize];
	_dMatrixACopy = new double[_iMatrixSize];
	
	_dVectorB = new double[_iMatrixHeight];
	_dVectorBCopy = new double[_iMatrixHeight];

	_dVectorX = new double[_iMatrixWidth];

	ZeroTheSystem();
	_pdWeights=NULL;  // debug notes
	_pcNotes=NULL;
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
void RobustSolverFOV::ZeroTheSystem()
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
bool RobustSolverFOV::AddCalibationConstraints(MosaicLayer* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex, bool bUseFiducials)
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
		else
		{
			//* 14 M6 = 0
			pdRowBegin[iFOVPos+6] = 2e7;
			pdRowBegin += _iMatrixWidth;
			_iCurrentRow++;

			//* 15 M7 = 0;
			pdRowBegin[iFOVPos+7] = 2e7;
			pdRowBegin += _iMatrixWidth;
			_iCurrentRow++;
		}
	}

	return(true);
}

// Add results for one Fov and Fov overlap
bool RobustSolverFOV::AddFovFovOvelapResults(FovFovOverlap* pOverlap)
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
bool RobustSolverFOV::AddCadFovOvelapResults(CadFovOverlap* pOverlap)
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
bool RobustSolverFOV::AddFidFovOvelapResults(FidFovOverlap* pOverlap)
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
	if (_bSaveMatrixCSV) 
	{
		// Save Matrix A
		ofstream of("C:\\Temp\\MatrixA.csv");
	
		for(unsigned int k=0; k<_iMatrixHeight; k++)
		{ 
			for(unsigned int j=0; j<_iMatrixWidth; j++)
			{
				of << _dMatrixA[j+k*_iMatrixWidth];
				if(j != _iMatrixWidth-1)
					of <<",";
			}
			of << std::endl;
		}
		of.close();

		// Save Matrix B
		of.open("C:\\Temp\\VectorB.csv");
		for(unsigned int k=0; k<_iMatrixHeight; k++)
		{ 
			of << _dVectorB[k] << std::endl;
		}
		of.close();

		// Save weights
		if (_pdWeights !=NULL)
		{
			of.open("C:\\Temp\\Weights.csv");
			for(unsigned int k=0; k<_iMatrixHeight; k++)
			{ 
				of << _pdWeights[k] << std::endl;
			}
			of.close();
		}

		// Save Notes
		if (_pcNotes!=NULL)
		{
			of.open("C:\\Temp\\Notes.csv");
			for(unsigned int k=0; k<_iMatrixHeight; k++)
			{ 
				of << _pcNotes[k] << std::endl;
			}
			of.close();
		}  
	}
	
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

	// for debug
	if (_bSaveMatrixCSV) 
	{
		// Save transposed Matrix A 
		ofstream of("C:\\Temp\\MatrixA_t.csv");		
		of << std::scientific;

		unsigned int ilines = _iMatrixHeight;
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
		of.open("C:\\Temp\\MatrixAOrdered.csv");		
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
		of.open("C:\\Temp\\VectorBOrdered.csv");
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
	}


	delete [] workspace;
	delete [] dCopyB;

	return(iMaxLength);
}

// Robust regression by Huber's "Algorithm H"
// Banded version
void RobustSolverFOV::SolveXAlgHB()
{

	::memcpy(_dMatrixACopy, _dMatrixA, _iMatrixSize*sizeof(double));
	::memcpy(_dVectorBCopy, _dVectorB, _iMatrixHeight*sizeof(double)); 

	// we built A in row order
	// the qr factorization method requires column order
	bool bRemoveEmptyRows = true;
	bRemoveEmptyRows = false;
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
ImgTransform RobustSolverFOV::GetResultTransform(
	unsigned int iLlluminationIndex,
	unsigned int iTriggerIndex,
	unsigned int iCameraIndex)  
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
bool RobustSolverFOV::MatchProjeciveTransform(const double pPara[12], double dTrans[3][3]) const
{
	int iNumX = 10;
	int iNumY = 10;
	// TODO TODO  hard coded image size
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
void RobustSolverFOV::OutputVectorXCSV(string filename) const
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
#pragma region Debug
// Debug
// Vector X output
void RobustSolverCM::OutputVectorXCSV(string filename) const
{
	ofstream of(filename.c_str());

	of << std::scientific;

	string line;
	unsigned int j;
	
	for(map<FovIndex, unsigned int>::iterator k=_pFovOrderMap->begin(); k!=_pFovOrderMap->end(); k++)
	{
		unsigned int iCamIndex( k->first.CameraIndex);
		unsigned int iIndexID( k->second );
		if (iCamIndex == 0)  // first camera (logical camera) is always numbered 0
		{
			of << "I_" << k->first.IlluminationIndex 
				<< "T_" << k->first.TriggerIndex 
				<< "C_" << k->first.CameraIndex
				<< ",";
			of << iIndexID;  // record index ID
			of << ",";
			for(j=0; j<_iNumParamsPerIndex; ++j)
			{
				if( j!=0 )
					of << ",";
				double d = _dVectorX[iIndexID * _iNumParamsPerIndex +j];
				of << d;
			}
			of <<  std::endl;
		}
	}
		
	for(unsigned int j=0; j<_iNumZTerms; ++j)
	{
		of << ",";

		double d = _dVectorX[_iMatrixWidth - _iNumZTerms + j];
		of << d;
	}
	of <<  std::endl;
	of.close();
}
#pragma endregion
RobustSolverCM::RobustSolverCM(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations,
		MosaicSet* pSet): 	RobustSolver( pFovOrderMap)
{
	// constructor, matrix size is changed, setup up all of the same variables
	_iNumBasisFunctions = 4;
	_iNumZTerms = 10;		// Z terms
	_iNumParamsPerIndex = 3;
	_pSet = pSet;
	unsigned int iTotalNumberOfTriggers = _pSet->GetMosaicTotalNumberOfTriggers();
	
	_iMatrixWidth = iTotalNumberOfTriggers * _iNumParamsPerIndex + _iNumZTerms;
	_iMatrixHeight = _iNumZTerms 
					+ iTotalNumberOfTriggers * _iNumParamsPerIndex
					+ 2*iMaxNumCorrelations;

	_iMatrixSize = _iMatrixWidth * _iMatrixHeight;

	_dMatrixA = new double[_iMatrixSize];
	_dMatrixACopy = new double[_iMatrixSize];
	
	_dVectorB = new double[_iMatrixHeight];
	_dVectorBCopy = new double[_iMatrixHeight];

	_dVectorX = new double[_iMatrixWidth];

	_iLengthNotes = 200;
	if(true)  // TODO make this selectable
	{
		_pdWeights = new double[_iMatrixHeight];
		_pcNotes	 = new char*[_iMatrixHeight];
		for(unsigned int i=0; i<_iMatrixHeight; i++)
			_pcNotes[i] = new char[_iLengthNotes];
	}


	ZeroTheSystem();

	// add global constraints
	// First add constraints on Z terms (which control allowed shape of the circuit board warp)
	//ConstrainZTerms();  //Don't do here, do after each reset
	// Next add the per trigger constraints
	//ConstrainPerTrig(pSet); //Don't do here, do after each reset
}
RobustSolverCM::~RobustSolverCM()
{
}

void RobustSolverCM::ZeroTheSystem()
{
	_iCurrentRow = 0;

	unsigned int i;
	for(i=0; i<_iMatrixSize; i++)
		_dMatrixA[i] = 0.0;

	for(i=0; i<_iMatrixHeight; i++)
	{
		_dVectorB[i] = 0.0;
		_pdWeights[i] = 0.0;
		sprintf_s(_pcNotes[i], _iLengthNotes, "");
	}

	for(i =0; i<_iMatrixWidth; i++)
		_dVectorX[i] = 0.0;
}

void RobustSolverCM::ConstrainZTerms()
{
	// board warp is modeled by 10 terms, a bi-variate cubic equation
	// the last columns of A describe board shape in Z
	// We'll give these a mild weight toward 0
	// EXCEPT for those that describe high order terms in Y
	// if there is only a single camera we're in trouble -- must rely on old means of measuring board height
	// if nCameras = 2 then we can only get a single range term (only y^0 terms)
	// if 3 cameras we get 2 range points and can est. y^0 and y^1 terms
	// if 4 cameras can add y^2 term
	// if 5 or more cameras (4 or more range points) then can add y^3 point
	unsigned int	nCameras(CountCameras());
	unsigned int ZfitTerms; // number of active terms
	double temp(Weights.wZConstrain);
	//*AlignAlgParam& arg(AlignAlgParam::instance());
	if (nCameras == 2)
		ZfitTerms = 4;  //
	else if (nCameras == 3)
		ZfitTerms = 7;
	else if (nCameras == 4)
		ZfitTerms = 9;
	else
		ZfitTerms = 10;
	
	double* Zterm; // pointer to location in A for setting Z
	for(unsigned int row(0); row< _iNumZTerms; ++row)
	{
		// adding Z control terms to the top of the array (for now), so row num is equal to row in A
		unsigned int Acol(_iMatrixWidth - _iNumZTerms + row);
		Zterm = &_dMatrixA[(row) * _iMatrixWidth];
		if (row < ZfitTerms)
			Zterm[Acol] = Weights.wZConstrain;
		else
			Zterm[Acol] = Weights.wZConstrainZero;
		//TODO add log of weights and debug notes
		_iCurrentRow++; // track how many rows of MatrixA have been used
				
	}
}
unsigned int RobustSolverCM::CountCameras()
{
	unsigned int minCameraNum(1000000);
	unsigned int maxCameraNum(0);
	// TODO WHAT IF STARTING CAMERA NUMBER VARIES BETWEEN ILLUM?
	for(map<FovIndex, unsigned int>::iterator k=_pFovOrderMap->begin(); k!=_pFovOrderMap->end(); k++)
	{
		if (minCameraNum > k->first.CameraIndex )
			minCameraNum = k->first.CameraIndex;
		if (maxCameraNum < k->first.CameraIndex )
			maxCameraNum = k->first.CameraIndex;
	}
	return (maxCameraNum - minCameraNum);

}
void RobustSolverCM::ConstrainPerTrig()
{
	// add constraints for each trigger  
	double dFovOriginU = 0.0;
	double dFovOriginV = 0.0;
	// for each trigger we add a constrain the xTrig, yTrig, theataTrig values to
	// their expected values
	// need to visit each trigger just once in the same order as _pFovOrderMap
	
	for(map<FovIndex, unsigned int>::iterator k=_pFovOrderMap->begin(); k!=_pFovOrderMap->end(); k++)
	{
		unsigned int iCamIndex( k->first.CameraIndex);
		if (iCamIndex == 0)  // first camera (logical camera) is always numbered 0
		{
			unsigned int indexID( k->second );
			unsigned int iIllumIndex( k->first.IlluminationIndex);
			unsigned int iTrigIndex( k->first.TriggerIndex);
			unsigned int iCols = _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->Columns();
			unsigned int iRows = _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->Rows();
			TransformCamModel camCal = 
				 _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetTransformCamCalibration();
			complexd xySensor;
			xySensor.r = htcorrp(iRows, iCols,
				dFovOriginU, dFovOriginV,
				_iNumBasisFunctions, _iNumBasisFunctions,
				(float *)camCal.S[CAL_ARRAY_X],  _iNumBasisFunctions);
			xySensor.i = htcorrp(iRows, iCols,
				dFovOriginU, dFovOriginV,
				_iNumBasisFunctions, _iNumBasisFunctions,
				(float *)camCal.S[CAL_ARRAY_Y],  _iNumBasisFunctions);
			
			// constrain x direction
			double* begin_pin_in_image_center(&_dMatrixA[(_iCurrentRow) * _iMatrixWidth]);
			unsigned int beginIndex( indexID * _iNumParamsPerIndex);
			begin_pin_in_image_center[beginIndex+0] = Weights.wXIndex;	// b VALUE IS NON_ZERO!!
			begin_pin_in_image_center[beginIndex+2] = -Weights.wXIndex * xySensor.i;
			_dVectorB[_iCurrentRow] = Weights.wXIndex * 
				(_pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetNominalTransform().GetItem(2)
				- xySensor.r);

			// TODO add logging

			// constrain yTrig value
			_iCurrentRow++;
			begin_pin_in_image_center= &_dMatrixA[(_iCurrentRow) * _iMatrixWidth];
			begin_pin_in_image_center[beginIndex+1] = Weights.wXIndex;	// b VALUE IS NON_ZERO!!
			begin_pin_in_image_center[beginIndex+2] = Weights.wXIndex * xySensor.r;
			_dVectorB[_iCurrentRow] = Weights.wXIndex * 
				(_pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetNominalTransform().GetItem(5)
				- xySensor.i);

			// constrain theata to zero
			_iCurrentRow++;
			begin_pin_in_image_center= &_dMatrixA[(_iCurrentRow) * _iMatrixWidth];
			begin_pin_in_image_center[beginIndex+2] = Weights.wXIndex;
			_iCurrentRow++;
		}
	}
}


bool RobustSolverCM::AddCalibationConstraints(MosaicLayer* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex, bool bUseFiducials)
{
	// there are no per FOV calib constraints in the camera model fit
	return true;
}
bool RobustSolverCM::AddFovFovOvelapResults(FovFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);

	// First Fov's information
	unsigned int iMosicIndexA = pOverlap->GetFirstMosaicImage()->Index();
	unsigned int iTrigIndexA = pOverlap->GetFirstTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetFirstCameraIndex();
	FovIndex index1(iMosicIndexA, iTrigIndexA, iCamIndexA); 
	unsigned int iFOVPosA = (*_pFovOrderMap)[index1] *_iNumParamsPerIndex;

	// Second Fov's information
	unsigned int iMosicIndexB = pOverlap->GetSecondMosaicImage()->Index();
	unsigned int iTrigIndexB = pOverlap->GetSecondTriggerIndex();
	unsigned int iCamIndexB = pOverlap->GetSecondCameraIndex();
	FovIndex index2(iMosicIndexB, iTrigIndexB, iCamIndexB); 
	unsigned int iFOVPosB = (*_pFovOrderMap)[index2] *_iNumParamsPerIndex;

	double* pdRow;

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
		// include 0 weight rows for now....
		//if(w <= 0) 
		//	continue;
		pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		// Get Centers of ROIs
		double rowImgA = (i->GetFirstRoi().FirstRow + i->GetFirstRoi().LastRow)/ 2.0;
		double colImgA = (i->GetFirstRoi().FirstColumn + i->GetFirstRoi().LastColumn)/ 2.0;

		double rowImgB = (i->GetSecondRoi().FirstRow + i->GetSecondRoi().LastRow)/ 2.0;
		double colImgB = (i->GetSecondRoi().FirstColumn + i->GetSecondRoi().LastColumn)/ 2.0;

		// Get offset
		double offsetRows = result.RowOffset;
		double offsetCols = result.ColOffset;
		rowImgB += offsetRows;
		colImgB += offsetCols;
		// Add a equataions 
		TransformCamModel camCalA = 
				 _pSet->GetLayer(iMosicIndexA)->GetImage(iCamIndexA, iTrigIndexA)->GetTransformCamCalibration();
		TransformCamModel camCalB = 
				 _pSet->GetLayer(iMosicIndexB)->GetImage(iCamIndexB, iTrigIndexB)->GetTransformCamCalibration();
		double xSensorA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.S[CAL_ARRAY_X],
						_iNumBasisFunctions);
		double ySensorA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.S[CAL_ARRAY_Y],
						_iNumBasisFunctions);
		double dxSensordzA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.dSdz[CAL_ARRAY_X],
						_iNumBasisFunctions);
		double dySensordzA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
						colImgA, rowImgA, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalA.dSdz[CAL_ARRAY_Y],
						_iNumBasisFunctions);
		double xSensorB = htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.S[CAL_ARRAY_X],
						_iNumBasisFunctions);
		double ySensorB = htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.S[CAL_ARRAY_Y],
						_iNumBasisFunctions);
		double dxSensordzB = htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.dSdz[CAL_ARRAY_X],
						_iNumBasisFunctions);
		double dySensordzB = htcorrp((int)camCalB.vMax, (int)camCalB.uMax,
						colImgB, rowImgB, 
						_iNumBasisFunctions, _iNumBasisFunctions,
						(float*)camCalB.dSdz[CAL_ARRAY_Y],
						_iNumBasisFunctions);
		pair<double, double> imgAOverlapCenter = i->GetFirstImg()->ImageToWorld(rowImgA,colImgA);
		pair<double, double> imgBOverlapCenter = i->GetSecondImg()->ImageToWorld(rowImgB,colImgB);
		double boardX = (imgAOverlapCenter.first + imgBOverlapCenter.first ) / 2.0;
		double boardY = (imgAOverlapCenter.second + imgBOverlapCenter.second) / 2.0;

		
		// Generate the Z poly terms
		double* Zpoly;
		Zpoly = new double[_iNumZTerms];
		// the z poly can be thought of as a partially populated 3x3 matrix:
		// 				X**0 * Y**0, X**1 * Y**0, X**2 * Y**0, X**3 * Y**0,
		//              X**0 * Y**1, X**1 * Y**1, X**2 * Y**1,      0,
		//              X**0 * Y**2, X**1 * Y**2,      0,           0,
		//              X**0 * Y**3       0,           0,           0

		Zpoly[0] = pow(boardX,0) * pow(boardY,0);
		Zpoly[1] = pow(boardX,1) * pow(boardY,0);
		Zpoly[2] = pow(boardX,2) * pow(boardY,0);
		Zpoly[3] = pow(boardX,3) * pow(boardY,0);
		Zpoly[4] = pow(boardX,0) * pow(boardY,1);
		Zpoly[5] = pow(boardX,1) * pow(boardY,1);
		Zpoly[6] = pow(boardX,2) * pow(boardY,1);
		Zpoly[7] = pow(boardX,0) * pow(boardY,2);
		Zpoly[8] = pow(boardX,1) * pow(boardY,2);
		Zpoly[9] = pow(boardX,0) * pow(boardY,3);

		pdRow[iFOVPosA] += w;
		pdRow[iFOVPosB] -= w;  // may be in same trigger number
		pdRow[iFOVPosA+2] += -ySensorA * w;
		pdRow[iFOVPosB+2] -= -ySensorB * w;
		// debug notes:
		// colImgA, rowImgA, colImgB, rowImgB, xSensorA, ySensorA, xSensorB, ySensorB
		for (unsigned int j(0); j < _iNumZTerms; j++)
			pdRow[_iMatrixWidth - _iNumZTerms + j] = Zpoly[j] * (dxSensordzA - dxSensordzB)* w;
		_dVectorB[_iCurrentRow] = -w * (xSensorA - xSensorB);
		sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FovFovCorr:I%d:T%d:C%d_I%d:T%d:C%d_%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e", 
			pOverlap->GetFirstMosaicImage()->Index(),
			pOverlap->GetFirstTriggerIndex(), pOverlap->GetFirstCameraIndex(),
			pOverlap->GetSecondMosaicImage()->Index(),
			pOverlap->GetSecondTriggerIndex(), pOverlap->GetSecondCameraIndex(),
			i->GetIndex(),
			colImgA, rowImgA, colImgB, rowImgB, xSensorA, ySensorA, xSensorB, ySensorB);
		_pdWeights[_iCurrentRow] = w;
	
		_iCurrentRow++;
		pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
		
		pdRow[iFOVPosA+1] += w;
		pdRow[iFOVPosB+1] -= w;  // may be in same trigger number
		pdRow[iFOVPosA+2] += xSensorA * w;
		pdRow[iFOVPosB+2] -= xSensorB * w;
		for (unsigned int j(0); j < _iNumZTerms; j++)
			pdRow[_iMatrixWidth - _iNumZTerms + j] = Zpoly[j] * (dySensordzA - dySensordzB)* w;
		_dVectorB[_iCurrentRow] = -w * (ySensorA - ySensorB);
		sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FovFovCorr:%d:I%d:C%d_%d:I%d:C%d_%d,%.2e,%.2e,%.4e,%.4e", 
			pOverlap->GetFirstMosaicImage()->Index(),
			pOverlap->GetFirstTriggerIndex(), pOverlap->GetFirstCameraIndex(),
			pOverlap->GetSecondMosaicImage()->Index(),
			pOverlap->GetSecondTriggerIndex(), pOverlap->GetSecondCameraIndex(),
			i->GetIndex(),
			i->GetCorrelationResult().CorrCoeff, i->GetCorrelationResult().AmbigScore,
			boardX,boardY);
		_pdWeights[_iCurrentRow] = w;
	
		_iCurrentRow++;
		

	
		delete [] Zpoly;
	}
	return( true );
}
bool RobustSolverCM::AddCadFovOvelapResults(CadFovOverlap* pOverlap)
{
	// there are no CAD apperture fits in the camera model fit
	// TODO TODO   is this true????????
	return( true );
}
bool RobustSolverCM::AddFidFovOvelapResults(FidFovOverlap* pOverlap)
{
	// Validation check for overlap
	if(!pOverlap->IsProcessed() || !pOverlap->IsGoodForSolver()) return(false);
	
	// Fov's information
	unsigned int iMosicIndexA= pOverlap->GetMosaicImage()->Index();
	unsigned int iTrigIndexA = pOverlap->GetTriggerIndex();
	unsigned int iCamIndexA = pOverlap->GetCameraIndex();
	FovIndex index(iMosicIndexA, iTrigIndexA, iCamIndexA); 
	unsigned int iFOVPosA = (*_pFovOrderMap)[index] *_iNumParamsPerIndex;

	double* pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;

	CorrelationPair* pPair = pOverlap->GetCoarsePair();

	// Validation check for correlation pair
	CorrelationResult result;
	bool bFlag = pPair->GetCorrelationResult(&result);
	if(!bFlag) 
		return(false);
	double w = Weights.CalWeight(pPair) * Weights.RelativeFidFovCamModWeight;
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
	rowImgA -= offsetRows;
	colImgA -= offsetCols;
	// Add a equataions 
	TransformCamModel camCalA = 
				_pSet->GetLayer(iMosicIndexA)->GetImage(iCamIndexA, iTrigIndexA)->GetTransformCamCalibration();
	double xSensorA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.S[CAL_ARRAY_X],
					_iNumBasisFunctions);
	double ySensorA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.S[CAL_ARRAY_Y],
					_iNumBasisFunctions);
	double dxSensordzA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.dSdz[CAL_ARRAY_X],
					_iNumBasisFunctions);
	double dySensordzA = htcorrp((int)camCalA.vMax, (int)camCalA.uMax,
					colImgA, rowImgA, 
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float*)camCalA.dSdz[CAL_ARRAY_Y],
					_iNumBasisFunctions);

	pair<double, double> imgAOverlapCenter = pPair->GetFirstImg()->ImageToWorld(rowImgA,colImgA);
	double boardX = imgAOverlapCenter.first;
	double boardY = imgAOverlapCenter.second;

	double* Zpoly;
	Zpoly = new double[_iNumZTerms];
	// the z poly can be thought of as a partially populated 4x4 matrix:
	// 				X**0 * Y**0, X**1 * Y**0, X**2 * Y**0, X**3 * Y**0,
	//              X**0 * Y**1, X**1 * Y**1, X**2 * Y**1,      0,
	//              X**0 * Y**2, X**1 * Y**2,      0,           0,
	//              X**0 * Y**3       0,           0,           0

	Zpoly[0] = pow(boardX,0) * pow(boardY,0);
	Zpoly[1] = pow(boardX,1) * pow(boardY,0);
	Zpoly[2] = pow(boardX,2) * pow(boardY,0);
	Zpoly[3] = pow(boardX,3) * pow(boardY,0);
	Zpoly[4] = pow(boardX,0) * pow(boardY,1);
	Zpoly[5] = pow(boardX,1) * pow(boardY,1);
	Zpoly[6] = pow(boardX,2) * pow(boardY,1);
	Zpoly[7] = pow(boardX,0) * pow(boardY,2);
	Zpoly[8] = pow(boardX,1) * pow(boardY,2);
	Zpoly[9] = pow(boardX,0) * pow(boardY,3);

	// X direction equations
	pdRow[iFOVPosA] += w;
	pdRow[iFOVPosA+2] += -ySensorA * w;
	for (unsigned int j(0); j < _iNumZTerms; j++)
			pdRow[_iMatrixWidth - _iNumZTerms + j] = Zpoly[j] * dxSensordzA * w;
	//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialXPos() - xSensorA);
	_dVectorB[_iCurrentRow] = w * (dFidRoiCenX - xSensorA);
	sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "FidCorr:%d:I%d:C%d,%.4e,%.4e,%.4e,%.4e,%.4e,%.4e", 
		pOverlap->GetMosaicImage()->Index(),
		pOverlap->GetTriggerIndex(), pOverlap->GetCameraIndex(),
		colImgA, rowImgA, xSensorA, ySensorA, pOverlap->GetFiducialXPos(),pOverlap->GetFiducialYPos() );
	_pdWeights[_iCurrentRow] = w;
	// Y direction equations
	_iCurrentRow++;
	pdRow = _dMatrixA + _iCurrentRow*_iMatrixWidth;
	pdRow[iFOVPosA+1] += w;
	pdRow[iFOVPosA+2] += +xSensorA * w;
	for (unsigned int j(0); j < _iNumZTerms; j++)
			pdRow[_iMatrixWidth - _iNumZTerms + j] = Zpoly[j] * dySensordzA * w;
	//_dVectorB[_iCurrentRow] = w * (pOverlap->GetFiducialYPos() - ySensorA);
	_dVectorB[_iCurrentRow] = w * (dFidRoiCenY - ySensorA);
	sprintf_s(_pcNotes[_iCurrentRow], _iLengthNotes, "Y_FidCorr:%d:I%d:C%d,%.2e,%.2e,%.4e,%.4e", 
		pOverlap->GetMosaicImage()->Index(),
		pOverlap->GetTriggerIndex(), pOverlap->GetCameraIndex(),
		pPair->GetCorrelationResult().CorrCoeff, pPair->GetCorrelationResult().AmbigScore,
		boardX,boardY);
	_pdWeights[_iCurrentRow] = w;
	
	_iCurrentRow++;

	delete [] Zpoly;
	return( true );
}
ImgTransform RobustSolverCM::GetResultTransform(
	unsigned int iIlluminationIndex,
	unsigned int iTriggerIndex,
	unsigned int iCameraIndex)  
{
	// Fov transform parameter begin position in column
	FovIndex fovIndex(iIlluminationIndex, iTriggerIndex, iCameraIndex); 
	unsigned int index = (*_pFovOrderMap)[fovIndex] *_iNumParamsPerIndex;
	unsigned int iImageCols = _pSet->GetLayer(iIlluminationIndex)->GetImage(iCameraIndex, iTriggerIndex)->Columns();
	unsigned int iImageRows = _pSet->GetLayer(iIlluminationIndex)->GetImage(iCameraIndex, iTriggerIndex)->Rows();
		
	double t[3][3];
	MatchProjeciveTransform(iIlluminationIndex, iTriggerIndex, iCameraIndex, t);
	
	ImgTransform trans(t);

	// add camModel transform calculation and load....
	// take set of points in image, set of points in CAD
	// calc matching transform (pixel to flattenend board surface)
	TransformCamModel tCamModel = _pSet->GetLayer(iIlluminationIndex)->GetImage(iCameraIndex,iTriggerIndex)->GetTransformCamModel();
	tCamModel.Reset();
	int iNumX = 11;
	int iNumY = 11;
	double startX = 0;
	double stopX = (double)(iImageRows - 1); // x direction is in + row direction
	double startY = 0;
	double stopY = (double)(iImageCols- 1); // y direction is in + col direction
	// use non-linear spacing to get better performance at edges
	double angleStepX = PI / (iNumX - 1.0);
	double angleStepY = PI / (iNumY - 1.0);
	
	POINTPIX* uv = new POINTPIX[iNumX*iNumY];
	POINT2D* xy =  new POINT2D[iNumX*iNumY];
	int i, j, k;
	//LOG.FireLogEntry(LogTypeSystem, "ResultFOVCamModel m %d, %d, %d",
	//	iIlluminationIndex, iTriggerIndex, iCameraIndex);
	for (j=0, i=0; j<iNumY; j++) {      /*uv.v value is image ROW which is actually parallel to CAD X */
	  for (k=0; k<iNumX; k++) {          /* uv.u value is image COL, parralel to CAD Y*/
			uv[i].v = (cos(-PI + angleStepX*k) + 1.0 )/2.0; // scaled 0 to 1
			uv[i].v *= stopX - startX;
	  		uv[i].u = (cos(-PI + angleStepY*j) + 1.0 )/2.0; // scaled 0 to 1
			uv[i].u *= stopY - startY;
		 POINT2D xyBoard;
		 Pix2Board(uv[i], fovIndex, &xyBoard); 
		// Now have position on flattened board,
		// need to transform to CAD postion using board2CAD
		// x position
		xy[i].x = _Board2CAD.GetItem(0)*xyBoard.x + _Board2CAD.GetItem(1)*xyBoard.y + _Board2CAD.GetItem(2);
		// y position
		xy[i].y = _Board2CAD.GetItem(3)*xyBoard.x + _Board2CAD.GetItem(4)*xyBoard.y + _Board2CAD.GetItem(5);
		i++;
	  }
	}
	tCamModel.CalcTransform(uv, xy, iNumX*iNumY);

	return(trans);
}

bool RobustSolverCM::MatchProjeciveTransform(	
	unsigned int iIlluminationIndex,
	unsigned int iTriggerIndex,
	unsigned int iCameraIndex, double dTrans[3][3]) 
{
	FovIndex fovIndex(iIlluminationIndex, iTriggerIndex, iCameraIndex); 
	unsigned int iImageCols = _pSet->GetLayer(iIlluminationIndex)->GetImage(iCameraIndex, iTriggerIndex)->Columns();
	unsigned int iImageRows = _pSet->GetLayer(iIlluminationIndex)->GetImage(iCameraIndex, iTriggerIndex)->Rows();
	
	unsigned int iNumX = 3;  // set to 3 uses corner and centers of edges (along with image center)
	unsigned int iNumY = 3;  // (the resulting fit is more accurate on the edges)
	double startX = 0;
	double stopX = (double)(iImageRows - 1); // x direction is in + row direction
	double startY = 0;
	double stopY = (double)(iImageCols- 1); // y direction is in + col direction
	// use non-linear spacing to get better performance at edges
	double angleStepX = PI / (iNumX - 1.0);
	double angleStepY = PI / (iNumY - 1.0);
	
	unsigned int m = iNumX*iNumY;
	complexd* p = new complexd[m];  // lsqrproj() uses complexd data type
	complexd* q = new complexd[m];
	complexd* resid = new complexd[m];
	POINT2D xyBoard;
	POINTPIX pix;  // data type needed by Pix2Board()
	// Create match pairs for projective transform
	// p will contain p.r = pixel row locations, p.i = pixel col locations
	// q.r will contain matching X (cad direction)
	// q.i will contain matching Y
	int i(0);
	for(unsigned int j=0; j<iNumY; j++)
	{
		for(unsigned int k=0; k<iNumX; k++)
		{
			p[i].r = (cos(-PI + angleStepX*k) + 1.0 )/2.0; // scaled 0 to 1
			p[i].r *= stopX - startX;
			p[i].i = (cos(-PI + angleStepY*j) + 1.0 )/2.0; // scaled 0 to 1
			p[i].i *= stopY - startY;
			// now find the matching point in CAD space
			// order is:
			// pixel -> xySensor -> xyBoard -> xyCAD
			// pixel to xyBoard
			pix.u = p[i].i;  // uv vs ri got ugly here
			pix.v = p[i].r;
			Pix2Board(pix, fovIndex, &xyBoard);
			//xyBoard to CAD
			POINT2D temp = warpxy(NUMBER_Z_BASIS_FUNCTIONS, 2*NUMBER_Z_BASIS_FUNCTIONS, NUMBER_Z_BASIS_FUNCTIONS,
								(double*)_zCoef, xyBoard, LAB_TO_CAD);
			// Now have position on flattened board,
			// need to transform to CAD postion using board2CAD
			// x position
			q[i].r = _Board2CAD.GetItem(0)*temp.x + _Board2CAD.GetItem(1)*temp.y + _Board2CAD.GetItem(2);
			// y position
			q[i].i = _Board2CAD.GetItem(3)*temp.x + _Board2CAD.GetItem(4)*temp.y + _Board2CAD.GetItem(5);
			i++;
		}
	}
	double RMS, dRcond;
	int iFlag = lsqrproj(m, p, q, 1, dTrans, &RMS, resid, &dRcond);
	if (_bVerboseLogging)
	{
		LOG.FireLogEntry(LogTypeSystem, "MatchProj index %d, %d, %d",iIlluminationIndex, iTriggerIndex, iCameraIndex); 
		LOG.FireLogEntry(LogTypeSystem, "MatchProj p %.4e, %.4e,    %.4e, %.4e,    %.4e, %.4e",
			p[0].i, p[0].r, p[1].i, p[1].r, p[2].i, p[2].r);
		LOG.FireLogEntry(LogTypeSystem, "MatchProj q %.4e, %.4e,    %.4e, %.4e,    %.4e, %.4e",
			q[0].i, q[0].r, q[1].i, q[1].r, q[2].i, q[2].r);
		LOG.FireLogEntry(LogTypeSystem, "dTrans \n %.4e, %.4e,%.4e, \n%.4e, %.4e,%.4e, \n%.4e, %.4e,%.4e",
			dTrans[0][0], dTrans[0][1], dTrans[0][2],
			dTrans[1][0], dTrans[1][1], dTrans[1][2],
			dTrans[2][0], dTrans[2][1], dTrans[2][2]);
		// NOTE: resid = q - fitValue = distorted pt - proj fit pt
	}
	if(iFlag != 0)
	{
		LOG.FireLogEntry(LogTypeError, "Failed to create matching projective transform");
	}
	

	if(iFlag != 0)
		return(false);
	else
		return(true);
}

// Robust regression by Huber's "Algorithm H"
// Banded version
void RobustSolverCM::SolveXAlgH()
{

	::memcpy(_dMatrixACopy, _dMatrixA, _iMatrixSize*sizeof(double));
	::memcpy(_dVectorBCopy, _dVectorB, _iMatrixHeight*sizeof(double)); 

	// we built A in row order
	// the qr factorization method requires column order
	bool bRemoveEmptyRows = false;
	//bRemoveEmptyRows = false;
	int* mb = new int[_iMatrixWidth];
	unsigned int iEmptyRows;
	unsigned int bw = ReorderAndTranspose(bRemoveEmptyRows, mb, &iEmptyRows);

	double*	resid = new double[_iMatrixHeight];
	double scaleparm = 0;
	double cond = 0;

	LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():BEGIN ALG_H");

	int algHRetVal = 
		alg_h(                // Robust regression by Huber's "Algorithm H".
			// Inputs //			

			_iMatrixHeight,             /* Number of equations */
			_iMatrixWidth,				/* Number of unknowns */
			_dMatrixA,				// System matrix
			_iMatrixHeight,			//AStride
			_dVectorB,			// Constant vector (not overwritten); must not coincide with x[] or resid[]. 

		   // Outputs //

			_dVectorX,            // Coefficient vector (overwritten with solution) 
			resid,				// Residuals b - A*x (pass null if not wanted).  If non-null, must not coincide with x[] or b[].
			&scaleparm,				// Scale parameter.  This is a robust measure of
									//  dispersion that corresponds to RMS error.  (Pass NULL if not wanted.)
			&cond					// Approximate reciprocal condition number.  (Pass NULL if not wanted.) 

			
		   /* Return values

			>=0 - normal return; value is # of iterations.
			 -1 - Incompatible dimensions (m mustn't be less than n)
			 -2 - singular system matrix
			 -3 - malloc failed.
			 -4 - Iteration limit reached (results may be tolerable)*/	
				  );

	// Save resid
	if (_bSaveMatrixCSV) 
	{
		ofstream of("C:\\Temp\\Resid.csv");
		for(unsigned int k=0; k<_iMatrixHeight; k++)
		{ 
			of << resid[k] << std::endl;
		}
		of.close();
	}

	if( algHRetVal<0 )
		LOG.FireLogEntry(LogTypeError, "RobustSolverCM::SolveXAlgH():alg_h returned value of %d", algHRetVal);

	//LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():FINISHED ALG_H");
	LOG.FireLogEntry(LogTypeSystem, "RobustSolverCM::SolveXAlgH():alg_h nIterations=%d, scaleparm=%f, cond=%f", algHRetVal, scaleparm, cond);

	delete [] resid;

	// copy board Z shape terms to the _zCoef array
	// First we'll put take VectorX values for Z and use it to populate Zpoly (a 4x4 array)
	// FORTRAN ORDER !!!!!
	_zCoef[0][0] = _dVectorX[_iMatrixWidth- _iNumZTerms + 0];
	_zCoef[1][0] = _dVectorX[_iMatrixWidth- _iNumZTerms + 1];
	_zCoef[2][0] = _dVectorX[_iMatrixWidth - _iNumZTerms + 2];
	_zCoef[3][0] = _dVectorX[_iMatrixWidth- _iNumZTerms + 3];
	
	_zCoef[0][1] = _dVectorX[_iMatrixWidth- _iNumZTerms + 4];
	_zCoef[1][1] = _dVectorX[_iMatrixWidth- _iNumZTerms + 5];
	_zCoef[2][1] = _dVectorX[_iMatrixWidth- _iNumZTerms + 6];
	_zCoef[3][1] = 0;
	
	_zCoef[0][2] = _dVectorX[_iMatrixWidth- _iNumZTerms + 7];
	_zCoef[1][2] = _dVectorX[_iMatrixWidth- _iNumZTerms + 8];
	_zCoef[2][2] = 0;
	_zCoef[3][2] = 0;

	_zCoef[0][3] = _dVectorX[_iMatrixWidth- _iNumZTerms + 9];
	_zCoef[1][3] = 0;
	_zCoef[2][3] = 0;
	_zCoef[3][3] = 0;
}

void  RobustSolverCM::Pix2Board(POINTPIX pix, FovIndex fovindex, POINT2D *xyBoard)
{
	// p2Board translates between a pixel location (for a given illumination, trigger and camera defined by index) to xyBoard location
	// remember that xyBoard is on a warped surface, so the xy distance between marks is likely to be less than the
	// CAD distance
	// Also remember that this board model has not been aligned to CAD
	unsigned int iIllumIndex( fovindex.IlluminationIndex );
	unsigned int iTrigIndex( fovindex.TriggerIndex );
	unsigned int iCamIndex( fovindex.CameraIndex );
	unsigned int iVectorXIndex = (*_pFovOrderMap)[fovindex] *_iNumParamsPerIndex;;
	TransformCamModel camCal = 
				 _pSet->GetLayer(iIllumIndex)->GetImage(iCamIndex, iTrigIndex)->GetTransformCamCalibration();
	complexd xySensor;
	xySensor.r = htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.S[CAL_ARRAY_X],  _iNumBasisFunctions);
	xySensor.i = htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.S[CAL_ARRAY_Y],  _iNumBasisFunctions);

	// use board transform to go to board x,y
	//xyBoard->r = cos(VectorX[index+2]) * xySensor.r  - sin(VectorX[index+2]) * xySensor.i  + VectorX[index+0];
	//xyBoard->i = sin(VectorX[index+2]) * xySensor.r  + cos(VectorX[index+2]) * xySensor.i  + VectorX[index+1];
	// try WITHOUT trig functions (small angle approx. used in fit, don't use different method here)
	xyBoard->x =        1         * xySensor.r  - _dVectorX[iVectorXIndex+2] * xySensor.i  + _dVectorX[iVectorXIndex+0];
	xyBoard->y = _dVectorX[iVectorXIndex+2] * xySensor.r  +        1         * xySensor.i  + _dVectorX[iVectorXIndex+1];

	// can now estimate Z   zCoef HAS FORTRAN ORDERING
	double Zestimate(0);
	for (unsigned int k(0); k < _iNumBasisFunctions; k++)
		for (unsigned int j(0); j < _iNumBasisFunctions; j++)
			Zestimate += _zCoef[j][k] * pow(xyBoard->x,(double)j) * pow(xyBoard->y,(double)k);

	// accurate Z value allows accurate calc of xySensor
	xySensor.r +=  Zestimate *
			htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.dSdz[CAL_ARRAY_X],  _iNumBasisFunctions);
	xySensor.i += Zestimate *
			htcorrp((int)camCal.vMax,(int)camCal.uMax,
					pix.u, pix.v,
					_iNumBasisFunctions, _iNumBasisFunctions,
					(float *)camCal.dSdz[CAL_ARRAY_Y],  _iNumBasisFunctions);
	// accurate xySensor allows accurate xyBrd calc
	//xyBoard->r = cos(VectorX[index+2]) * xySensor.r  - sin(VectorX[index+2]) * xySensor.i  + VectorX[index+0];
	//xyBoard->i = sin(VectorX[index+2]) * xySensor.r  + cos(VectorX[index+2]) * xySensor.i  + VectorX[index+1];
	// try WITHOUT trig functions (small angle approx. used in fit, don't use different method here)
	xyBoard->x =               1 * xySensor.r             -  _dVectorX[iVectorXIndex+2] * xySensor.i  + _dVectorX[iVectorXIndex+0];
	xyBoard->y = _dVectorX[iVectorXIndex+2] * xySensor.r  +                1 * xySensor.i             + _dVectorX[iVectorXIndex+1];

}
void RobustSolverCM::FlattenFiducials(PanelFiducialResultsSet* fiducialSet)
{
	// translate the fiducial locations on the board down to a nominal height plane
	// uses warpxy.c to do this translation
	// then do a least sq. fit to do an affine xform between board and CAD 

	// count the number fiducial overlaps in the data set 
	unsigned int nFidOverlaps(0);
	int iNumPhyFid = fiducialSet->Size();
	for(int i=0; i<iNumPhyFid; i++)
	{
		list<FidFovOverlap*>* pResults = fiducialSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
			nFidOverlaps++;
	}
	POINT2D* fidFlat2D = new POINT2D[nFidOverlaps]; // fiducial location on flattened board
	POINT2D* fidCAD2D = new	POINT2D[nFidOverlaps];  // true CAD location of fiducial
	double resetTransformValues[3][3] =  {{1.0,0,0},{0,1.0,0},{0,0,1}};
	_Board2CAD.SetMatrix( resetTransformValues );
	unsigned int nGoodFids(0);  // <= total number of fids
	// Now add the fiducials to the two arrays created above
	for(int i=0; i<iNumPhyFid; i++)
	{
		list<FidFovOverlap*>* pResults = fiducialSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
		{
			bool fidOK;
			if ( (*j)->GetFiducialSearchMethod() == FIDVSFINDER)
				fidOK = ( (*j)->GetCoarsePair()->GetCorrelationResult().CorrCoeff 
					- (*j)->GetCoarsePair()->GetCorrelationResult().AmbigScore * 0.5 ) > 0; // MAGIC NUMBER !!!!!!!!!!!!!!!!!!
			else  // must be regoff  //TODO make these magic numbers variable
				fidOK = ( (*j)->GetCoarsePair()->GetCorrelationResult().CorrCoeff 
					- (*j)->GetCoarsePair()->GetCorrelationResult().AmbigScore * 0.5 ) > 0; // MAGIC NUMBER !!!!!!!!!!!!!!!!!!
			if (fidOK)
			{
				// add CAD locations
				double rowImgB = ((*j)->GetCoarsePair()->GetSecondRoi().FirstRow + (*j)->GetCoarsePair()->GetSecondRoi().LastRow)/ 2.0;
				double colImgB = ((*j)->GetCoarsePair()->GetSecondRoi().FirstColumn + (*j)->GetCoarsePair()->GetSecondRoi().LastColumn)/ 2.0;
				double dFidRoiCenX, dFidRoiCenY; // ROI center of fiducial image (not fiducail center or image center) in world space
				(*j)->GetFidImage()->ImageToWorld(rowImgB, colImgB, &dFidRoiCenX, &dFidRoiCenY);
				fidCAD2D [nGoodFids].x = dFidRoiCenX;
				fidCAD2D [nGoodFids].y = dFidRoiCenY;
				//fidCAD2D [nGoodFids].x = (*j)->GetFiducialXPos();
				//fidCAD2D [nGoodFids].y = (*j)->GetFiducialYPos();

				// now find board locations
				// given pixel location and S, dSdz find xySensor (assume Z = 0 to begin with)
				// Offset of unclipped center and clipped center (copied from DoImgFidCorrelationResult() )
				// TODO TODO Are these values ever != 0 ???? (firstcolumn !=0) ???
				double dCenOffsetX = ( (*j)->GetCoarsePair()->GetSecondRoi().Columns() - 1.0 )/2.0 - 
					((*j)->GetCoarsePair()->GetSecondRoi().FirstColumn + (*j)->GetCoarsePair()->GetSecondRoi().LastColumn)/2.0;
				double dCenOffsetY = ( (*j)->GetCoarsePair()->GetSecondRoi().Rows() - 1.0 )/2.0 - 
					((*j)->GetCoarsePair()->GetSecondRoi().FirstRow + (*j)->GetCoarsePair()->GetSecondRoi().LastRow)/2.0;
				
				// Unclipped image patch (first image of overlap) center in the overlap + alignment offset
				//double colImg = (overlap._firstOverlap.FirstColumn  + overlap._firstOverlap.LastColumn)  / 2.0 + dCenOffsetX;
				//double rowImg = (overlap._firstOverlap.FirstRow     + overlap._firstOverlap.LastRow)     / 2.0 + dCenOffsetY;
				double colImg = ( (*j)->GetCoarsePair()->GetFirstRoi().FirstColumn + 
					(*j)->GetCoarsePair()->GetFirstRoi().LastColumn )/2.0;//  + dCenOffsetX;
				double rowImg = ( (*j)->GetCoarsePair()->GetFirstRoi().FirstRow +
					(*j)->GetCoarsePair()->GetFirstRoi().LastRow )/2.0; //  + dCenOffsetY;
				double regoffCol = (*j)->GetCoarsePair()->GetCorrelationResult().ColOffset;
				double regoffRow = (*j)->GetCoarsePair()->GetCorrelationResult().RowOffset;
				double rowPeak( rowImg - regoffRow );
				double colPeak( colImg - regoffCol ); 
				POINT2D xyBoard;
				POINTPIX pix;
				pix.u = colPeak;  // Pix2Board uses u -> i, v -> r
				pix.v = rowPeak;
				FovIndex fovIndex( (*j)->GetMosaicImage()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex() );
				Pix2Board(pix, fovIndex, &xyBoard);
					
				// brdFlat from warpxy
				POINT2D temp;
				temp = warpxy(NUMBER_Z_BASIS_FUNCTIONS, 2*NUMBER_Z_BASIS_FUNCTIONS, NUMBER_Z_BASIS_FUNCTIONS,
									(double*)_zCoef, xyBoard, LAB_TO_CAD);
				fidFlat2D[nGoodFids].x = temp.x;
				fidFlat2D[nGoodFids].y = temp.y;
				nGoodFids++;
			}
		}
	}
	// Rows of FidFitA contain the following values:
	//  0		Constrain scale X ~ 1.0,  m0 ~ 1
	//  1		Constrain scale Y ~ 1.0,  m4 ~ 1
	//  2		Constrain scale x = scale y,   m0-m4 ~ 0
	//  3		Constrain m1 + m3  ~ 0  keep rectangular
	//  4		Constrain m1 ~ 0  small angle
	//  5		Constrain m3 ~ 0  small angle
	//  6		Constrain m2 ~ 0  small lateral shift
	//  7       Constrain m5 ~ 0  small lateral shift
	// 
	// Total 8 constraints
	int nContraints = 8;
		
	// Create A and b matrices, these are small so we can create/delete them without using big blocks of memory
	double		*FidFitA;
	double		*FidFitX;
	double		*FidFitb;
	double		*resid;
	unsigned int FidFitCols = 6;
	unsigned int FidFitRows = nGoodFids*2 + 8;
	int SizeofFidFitA = FidFitCols * FidFitRows;
	FidFitA = new double[SizeofFidFitA];
	FidFitX = new double[FidFitCols];
	FidFitb = new double[FidFitRows];
	resid   = new double[FidFitRows];
	// zeros the system
	for(size_t s(0); s<SizeofFidFitA; ++s)
		FidFitA[s] = 0.0;
	for (size_t s(0); s<FidFitRows; ++s)
		FidFitb[s] = 0.0;
	for (size_t s(0); s<FidFitCols; ++s)
		FidFitX[s] = 0.0;
	// add constraints
	FidFitb[0] = 1.0 * Weights.wFidFlatBoardScale;
	FidFitA[0*FidFitCols + 0] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitb[1] = 1.0 * Weights.wFidFlatBoardScale;
	FidFitA[1*FidFitCols + 4] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitb[2] = 0.0 * Weights.wFidFlatBoardScale;
	FidFitA[2*FidFitCols + 0] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitA[2*FidFitCols + 4] = -1.0 * Weights.wFidFlatBoardScale;
	FidFitb[3] = 0.0 * Weights.wFidFlatBoardScale;
	FidFitA[3*FidFitCols + 1] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitA[3*FidFitCols + 3] =  1.0 * Weights.wFidFlatBoardScale;
	FidFitb[4] = 0.0 * Weights.wFidFlatBoardRotation;
	FidFitA[4*FidFitCols + 1] =  1.0 * Weights.wFidFlatBoardRotation;
	FidFitb[5] = 0.0 * Weights.wFidFlatBoardRotation;
	FidFitA[5*FidFitCols + 3] =  1.0 * Weights.wFidFlatBoardRotation;
	FidFitb[6] = 0.0 * Weights.wFidFlatFiducialLateralShift;
	FidFitA[6*FidFitCols + 2] =  1.0 * Weights.wFidFlatFiducialLateralShift;
	FidFitb[7] = 0.0 * Weights.wFidFlatFiducialLateralShift;
	FidFitA[7*FidFitCols + 5] =  1.0 * Weights.wFidFlatFiducialLateralShift;
	// add the actual fiducial locations
	int AStart(0);
	for (unsigned int j(0); j< nGoodFids; j++)
	{
		FidFitb[8+j*2+0] = fidCAD2D[j].x * Weights.wFidFlatFlattenFiducial;
		FidFitb[8+j*2+1] = fidCAD2D[j].y * Weights.wFidFlatFlattenFiducial;
		AStart = (8+j*2+0)*FidFitCols;
		FidFitA[AStart + 0] = fidFlat2D[j].x * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 1] = fidFlat2D[j].y * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 2] = Weights.wFidFlatFlattenFiducial;
		AStart = (8+j*2+1)*FidFitCols;
		FidFitA[AStart + 3] = fidFlat2D[j].x * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 4] = fidFlat2D[j].y * Weights.wFidFlatFlattenFiducial;
		FidFitA[AStart + 5] = Weights.wFidFlatFlattenFiducial;
	}
	// Note: above equations do not weight the fiducial finds by corrcoef or ambig
	// ('bad' fids are already removed from the list)
	// As the number of fids is small each one one will have a great deal of leverage
	// therefore it's hard to de-weight poorer fits
	if (_bVerboseLogging)
	{
		for (unsigned int row(0); row < FidFitRows; row++)
			LOG.FireLogEntry(LogTypeSystem, "%d, %.6e, %.6e, %.6e, %.6e, %.6e, %.6e,       \t %.6e,        ",row, 
				FidFitA[row*6], FidFitA[row*6+1], FidFitA[row*6+2], FidFitA[row*6+3], 
				FidFitA[row*6+4], FidFitA[row*6+5], FidFitb[row] );
	}
	// equations are done, now solve

	LstSqFit(FidFitA, FidFitRows, FidFitCols, FidFitb, FidFitX, resid);
	if (_bVerboseLogging)
		for(unsigned int row(0); row<FidFitRows; ++row)
			LOG.FireLogEntry(LogTypeSystem, "Resids %d, %.4f",row, resid[row]);
	double newBoard2CAD[3][3] = {{ FidFitX[0], FidFitX[1], FidFitX[2]}, {FidFitX[3], FidFitX[4], FidFitX[5]}, {0,0,1}};
	_Board2CAD.SetMatrix(newBoard2CAD);
	
	delete [] FidFitA;
	delete [] FidFitb;
	delete [] FidFitX;
	delete [] resid;
		
	// parse results
	// log results for debug
	

	if (_bVerboseLogging)
		LOG.FireLogEntry(LogTypeSystem, "Flatten Fiducial _Board2CAD %.5e,%.5e,%.5e,   %.5e,%.5e,%.5e    ", 
			_Board2CAD.GetItem(0), _Board2CAD.GetItem(1), _Board2CAD.GetItem(2), _Board2CAD.GetItem(3), _Board2CAD.GetItem(4), _Board2CAD.GetItem(5));
		
	delete [] fidFlat2D;
	delete [] fidCAD2D;


}

void RobustSolverCM::LstSqFit(double *A, unsigned int nRows, unsigned int nCols, double *b, double *X, double *resid)
{
	// a very generic least square solver wrapper
	// takes input arrays in 'C' format and does the requried steps to call Eric's qr tools
	// note that b is NOT overwritten (unlike a raw call to qr)
	// resid is a vector of b-Ax values
	double		*workspace;
	double 		*bCopy;
	bCopy = new double[nRows];
	int SizeofFidFitA = nCols * nRows;
	workspace = new double[SizeofFidFitA];
	
		
	// transpose for Eric's code
	for(unsigned int row(0); row<nRows; ++row)
	{
		bCopy[row] = b[row];
		for(unsigned int col(0); col<nCols; ++col)
			workspace[col*nRows+row] = A[row*nCols+col];
	}
	// solve copied from SolveX()
	double*	sigma = new double[nCols];

	// perform factorization of system A = QR
	int qrdRet = 
		qrd(
				workspace,    /* System matrix, m-by-n */
				sigma,      /* Diagonals of R (caller reserves n elements) */
				nRows,      /* Number of rows in system matrix */
				nRows,		/* Spacing between columns in system matrix */
				nCols ); /* Number of columns in system matrix */

							/* Return values

								 0 - Normal completion.
								 1 - Matrix was of incompatible dimensions.
								 2 - Singular system matrix. */
	//LOG.FireLogEntry(LogTypeSystem, "qrdRet %d", qrdRet);
	//for (unsigned int row(0); row < nCols; row++)
	//		LOG.FireLogEntry(LogTypeSystem, "sigma %d, %.3e",row, sigma[row]);
   double condition =
	   rcond(               /* Reciprocal condition number estimator */
			   nCols,    /* Number of unknowns */
			   workspace,     /* QR factorization returned from qrd() */
			   nRows,       /* Spacing between columns of qr[] */
			   sigma        /* R diagonals from qrd() */
			);
   //LOG.FireLogEntry(LogTypeSystem, "condition %f", condition);

	qrsolv (
				workspace,	/* Factored system matrix, from qrd() */
				sigma,		/* Diagonal of R, from qrd() */
				&bCopy[0],	/* Constant vector, overwritten by solution */
				nRows,		/* Number of rows in system matrix */
				nRows,		/* Spacing between columns in system matrix */
				nCols,	/* Number of columns in system matrix */
				1     );	/*	0 - premultiply  b  by Q
								1 - premultiply  b  by inv(QR)
								2 - premultiply  b  by QR
								3 - premultiply  b  by inv(R^T R)
								4 - premultiply  b  by Q inv(R^T)
								5 - premultiply  b  by R
								6 - premultiply  b  by inv(R)

								 qrsolv() always returns 0 */


	for (unsigned int j(0); j<nCols; ++j)
		X[j] = bCopy[j];
	
	for(unsigned int row(0); row<nRows; ++row)
	{
		resid[row] = b[row] ;
		for (unsigned int col(0); col<nCols; ++col)
		  resid[row] -=  X[col]*A[row*nCols+col];
	}
	delete [] sigma;
	delete [] workspace;
	delete [] bCopy;
}
