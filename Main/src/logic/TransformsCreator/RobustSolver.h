#pragma once

#include "MosaicImage.h"
#include "OverlapDefines.h"

class FovIndex
{
public:
	FovIndex(
		unsigned int iIllumIndex,
		unsigned int iTrigIndex,
		unsigned int iCamIndex)
	{
		IlluminationIndex = iIllumIndex;
		TriggerIndex = iTrigIndex;
		CameraIndex = iCamIndex;
	}

	unsigned int IlluminationIndex;
	unsigned int TriggerIndex;
	unsigned int CameraIndex;
};

bool operator<(const FovIndex& a, const FovIndex& b);
bool operator>(const FovIndex& a, const FovIndex& b);

// Only support single thread so far
class RobustSolver
{
public:
	RobustSolver(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations, 
		bool bProjectiveTrans = false);
	~RobustSolver(void);

	bool AddCalibationConstraints(MosaicImage* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex);
	bool AddFovFovOvelapResults(FovFovOverlap* pOverlap);
	bool AddCadFovOvelapResults(CadFovOverlap* pOverlap);
	bool AddFidFovOvelapResults(FidFovOverlap* pOverlap);
	void SolveXAlgHB();
	
	ImgTransform RobustSolver::GetResultTransform(
		unsigned int iLlluminationIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex) const;

protected:
	void ZeroTheSystem();
	unsigned int ReorderAndTranspose(bool bRemoveEmptyRows, int* piCounts, unsigned int* piEmptyRows);
	bool MatchProjeciveTransform(const double pPara[12], double dTrans[3][3]) const;

private:
	map<FovIndex, unsigned int>* _pFovOrderMap;
	bool			_bProjectiveTrans;	
	unsigned int	_iMaxNumCorrelations;
	
	unsigned int	_iNumFovs;
	
	unsigned int	_iNumCalibConstrains;
	unsigned int	_iNumParamsPerFov;
	unsigned int	_iMatrixWidth;
	unsigned int	_iMatrixHeight;
	unsigned int	_iMatrixSize;

	unsigned int	_iCurrentRow;

	double*			_dMatrixA;
	double*			_dMatrixACopy;
	double*			_dVectorB;
	double*			_dVectorBCopy;
	double*			_dVectorX;
};

