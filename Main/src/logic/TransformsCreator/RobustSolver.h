#pragma once

#include "MosaicImage.h"
#include "OverlapManager.h"

class RobustSolver
{
public:
	RobustSolver(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iNumCorrelations, 
		bool bProjectiveTrans = false);
	~RobustSolver(void);

	bool AddCalibationConstraints(MosaicImage* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex);

protected:
	void ZeroTheSystem();

private:
	map<FovIndex, unsigned int>* _pFovOrderMap;
	bool			_bProjectiveTrans;	
	unsigned int	_iNumCorrelations;
	
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

