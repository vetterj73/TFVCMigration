
#pragma once

#include "RobustSolver.h"

// camera model distortion array horizontal direction is index 0 == CAD direction Y
// index 1 == CAD direction X
#define CAL_ARRAY_X			1
#define CAL_ARRAY_Y			0


class RobustSolverCM: public RobustSolver
{
public:
	RobustSolverCM(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations,
		MosaicSet* pSet);

	virtual ~RobustSolverCM(void);

	virtual bool AddAllLooseConstraints(
		bool bPinPanelWithCalibration=false, 
		bool bUseNominalTransform=true);

	virtual bool AddPanelEdgeContraints(
		MosaicLayer* pLayer, unsigned int iCamIndex, unsigned int iTrigIndex, 
		double dXOffset, double dSlope, bool bSlopeOnly=false);
	virtual bool AddFovFovOvelapResults(FovFovOverlap* pOverlap);
	virtual bool AddCadFovOvelapResults(CadFovOverlap* pOverlap);
	virtual bool AddFidFovOvelapResults(FidFovOverlap* pOverlap);
	virtual bool AddInputFidLocations(FiducialLocation* pLoc){return true;};

	virtual ImgTransform GetResultTransform(
		unsigned int iLlluminationIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex);
	virtual void OutputVectorXCSV(string filename) const;
	virtual void Reset() {ZeroTheSystem();};
	virtual void SolveXAlgH();
	virtual void			FlattenFiducials(PanelFiducialResultsSet* fiducialSet);

	bool GetPanelHeight(unsigned int iDeviceIndex, double pZCoef[16]);

protected:
	void			ReorderAndTranspose(bool bRemoveEmptyRows);
	unsigned int	ColumnZTerm(unsigned int term, unsigned int deviceNum);
	virtual void	ZeroTheSystem();
	virtual void	ConstrainZTerms();
	virtual void	ConstrainPerTrig();
	
	unsigned int	CountCameras();
	virtual void	Pix2Board(POINTPIX pix, FovIndex index, POINT2D *xyBoard);
	void			LstSqFit(double *FidFitA, unsigned int FidFitRows, unsigned int FidFitCols, double *FidFitb, double *FidFitX, double *resid);
	bool MatchProjeciveTransform(	
		unsigned int iLayerIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex, 
		double dTrans[3][3]) ;
	
	unsigned int	_iNumParamsPerIndex;
	unsigned int	_iNumZTerms;
	unsigned int	_iStartColZTerms;
	unsigned int	_iNumBasisFunctions;
	unsigned int	_iLengthNotes;
	unsigned int	_iNumFids;
	unsigned int	_iNumCameras; // Is number of cameras the same for all devices?,  Does the first camera always start at 0?
	unsigned int	_iNumDevices;
	unsigned int	_iTotalNumberOfTriggers;
	unsigned int	_iNumCalDriftTerms;
	unsigned int	_iMaxNumCorrelations;
	unsigned int	_iMatrixALastRowUsed;  // 

	unsigned int	_iIterationNumber;
	unsigned int	_iMaxIterations;

	ImgTransform	_Board2CAD;  // used by flatten fiducial to map from warped board XY to flat CAD XY
	double			_zCoef[MAX_NUMBER_DEVICES][NUMBER_Z_BASIS_FUNCTIONS][NUMBER_Z_BASIS_FUNCTIONS];
	// above uses a define !!  tied to _iNumZTerms wich is a variable !!
	

};