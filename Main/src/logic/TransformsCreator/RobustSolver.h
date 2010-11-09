#pragma once

#include "MosaicImage.h"
#include "OverlapManager.h"

class RobustSolver
{
public:
	RobustSolver(unsigned int iNumFovs, unsigned int iNumEquations, bool bProjectiveTrans = false);
	~RobustSolver(void);

	void AddCalibationConstraints(MosaicImage* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex);

protected:
	void ZeroTheSystem();

private:
	bool			_bProjectiveTrans;	
	unsigned int	_iNumFovs;
	unsigned int	_iNumEquations;
	
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

