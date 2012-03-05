/* 
	The banded robust least square solver class
*/

#pragma once

#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "OverlapDefines.h"
#include "FiducialResult.h"
#include "Logger.h"
#include "coord.h"
#include <direct.h> //_mkdir

#define NUMBER_Z_BASIS_FUNCTIONS 4
#define PI        (3.141592653589793)


using namespace MosaicDM;
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
	FovIndex()
	{
		IlluminationIndex = 0;
		TriggerIndex = 0;
		CameraIndex = 0;
	}
	unsigned int IlluminationIndex;
	unsigned int TriggerIndex;   // trigger order within Illumination
	unsigned int CameraIndex;	// camera order within trigger
	// TODO does camera index always start with 0?
};

bool operator<(const FovIndex& a, const FovIndex& b);
bool operator>(const FovIndex& a, const FovIndex& b);

// Only support single thread so far
class RobustSolver
{
public:
	RobustSolver(
		map<FovIndex, unsigned int>* pFovOrderMap);
	virtual ~RobustSolver(void);

	virtual bool AddCalibationConstraints(MosaicLayer* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex, bool bUseFiducials)=0;
	virtual bool AddFovFovOvelapResults(FovFovOverlap* pOverlap)=0;
	virtual bool AddCadFovOvelapResults(CadFovOverlap* pOverlap)=0;
	virtual bool AddFidFovOvelapResults(FidFovOverlap* pOverlap)=0;
	virtual void SolveXAlgH()=0;
	
	virtual ImgTransform GetResultTransform(
		unsigned int iLlluminationIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex) =0;

	virtual void			ConstrainZTerms()=0;
	virtual void			ConstrainPerTrig()=0;

	// Debug
	virtual void OutputVectorXCSV(string filename) const=0;
	virtual void Reset()=0;
	virtual void FlattenFiducials(PanelFiducialResultsSet* fiducialSet)=0;
	int				iFileSaveIndex;  // mark output files with this number

protected:
	virtual void ZeroTheSystem() =0;
	unsigned int ReorderAndTranspose(bool bRemoveEmptyRows, int* piCounts, unsigned int* piEmptyRows);
	//virtual bool MatchProjeciveTransform(const double pPara[12], double dTrans[3][3]) const=0;

//private:   // protected so they can be inherited ?!
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
	double*				_pdWeights;  // debug notes
	char**				_pcNotes;

	bool			_bSaveMatrixCSV;
	bool			_bVerboseLogging;

	
};




