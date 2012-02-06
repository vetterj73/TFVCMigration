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
	~RobustSolver(void);

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
};


class RobustSolverFOV: public RobustSolver
{
public:
	RobustSolverFOV(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations, 
		bool bProjectiveTrans);

	~RobustSolverFOV(void);

	bool AddCalibationConstraints(MosaicLayer* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex, bool bUseFiducials);
	bool AddFovFovOvelapResults(FovFovOverlap* pOverlap);
	bool AddCadFovOvelapResults(CadFovOverlap* pOverlap);
	bool AddFidFovOvelapResults(FidFovOverlap* pOverlap);

	void			ConstrainZTerms(){};
	void			ConstrainPerTrig(){};
	void			FlattenFiducials(PanelFiducialResultsSet* fiducialSet){};

	ImgTransform GetResultTransform(
		unsigned int iLlluminationIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex);
	void OutputVectorXCSV(string filename) const;
	void Reset() {ZeroTheSystem();};
	void SolveXAlgH() {SolveXAlgHB();};
	

protected:
	void ZeroTheSystem();
	void SolveXAlgHB();
private:
	bool MatchProjeciveTransform(const double pPara[12], double dTrans[3][3]) const;

};

class RobustSolverCM: public RobustSolver
{
public:
	RobustSolverCM(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations,
		MosaicSet* pSet);

	~RobustSolverCM(void);

	bool AddCalibationConstraints(MosaicLayer* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex, bool bUseFiducials);
	bool AddFovFovOvelapResults(FovFovOverlap* pOverlap);
	bool AddCadFovOvelapResults(CadFovOverlap* pOverlap);
	bool AddFidFovOvelapResults(FidFovOverlap* pOverlap);

	ImgTransform GetResultTransform(
		unsigned int iLlluminationIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex) ;
	void OutputVectorXCSV(string filename) const;
	void			ConstrainZTerms();
	void			ConstrainPerTrig();
	void Reset() {ZeroTheSystem();};
	void SolveXAlgH();
	void			FlattenFiducials(PanelFiducialResultsSet* fiducialSet);

	

protected:
	void ZeroTheSystem();
	

private:
	MosaicSet*		_pSet;
	unsigned int	CountCameras();
	void			Pix2Board(POINTPIX pix, FovIndex index, POINT2D *xyBoard);
	void			LstSqFit(double *FidFitA, unsigned int FidFitRows, unsigned int FidFitCols, double *FidFitb, double *FidFitX, double *resid);
	bool MatchProjeciveTransform(	
		unsigned int iIlluminationIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex, 
		double dTrans[3][3]) ;
	unsigned int	_iNumParamsPerIndex;
	unsigned int	_iNumZTerms;
	unsigned int	_iNumBasisFunctions;
	unsigned int	_iLengthNotes;
	unsigned int	_iNumFids;

	ImgTransform	_Board2CAD;  // used by flatten fiducial to map from warped board XY to flat CAD XY
	double			_zCoef[NUMBER_Z_BASIS_FUNCTIONS][NUMBER_Z_BASIS_FUNCTIONS];
	// above uses a define !!  tied to _iNumZTerms wich is a variable !!
	

};