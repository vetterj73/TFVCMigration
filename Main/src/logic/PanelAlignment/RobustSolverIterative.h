#pragma once

#include "RobustSolverCM.h"

struct fitInfo {
	int				fitType;		// 0 == not used, 1 == fid to fov, 2 == fov to fov, 3 == board edge, 4 == board edge slope only
	FovIndex		fovIndexA;
	FovIndex		fovIndexB;		// not used in fid to fov 
	//int				orderedTrigIndexA; // position in overall trigger sequence
	//int				orderedTrigIndexB; // actuall encoded in the colInMatrix[A,B]
	double			boardX;			// board xy pos est for Z calc
	double			boardY;
	double			w;				// fit weight
	unsigned int	rowInMatrix;
	unsigned int	colInMatrixA;	// col in matrix for fov A
	unsigned int	colInMatrixB;	// col in matrix for fov B
	double			xSensorA;
	double			ySensorA;
	double			xSensorB;
	double			ySensorB;
	double			dxSensordzA;
	double			dySensordzA;
	double			dxSensordzB;
	double			dySensordzB;
	double			dFidRoiCenX;
	double			dFidRoiCenY;
	fitInfo() {};
};

class RobustSolverIterative: public RobustSolverCM
{
public:
	RobustSolverIterative(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations,
		MosaicSet* pSet);

	virtual ~RobustSolverIterative(void);
	void	SolveXAlgH();
	
	bool AddAllLooseConstraints(
		bool bPinPanelWithCalibration=false, 
		bool bUseNominalTransform=true);

	//bool AddCalibationConstraints(MosaicLayer* pMosaic, unsigned int iCamIndex, unsigned int iTrigIndex, bool bUseFiducials);
	bool AddPanelEdgeContraints(
		MosaicLayer* pLayer, unsigned int iCamIndex, unsigned int iTrigIndex, 
		double dXOffset, double dSlope, bool bSlopeOnly=false);
	bool AddFovFovOvelapResults(FovFovOverlap* pOverlap);
	bool AddCadFovOvelapResults(CadFovOverlap* pOverlap);
	bool AddFidFovOvelapResults(FidFovOverlap* pOverlap);
	bool AddInputFidLocations(FiducialLocation* pLoc){return true;};
	void OutputVectorXCSV(string filename) const;

	void SetPinPanelWithCalibration(bool bVal) { _bPinPanelWithCalibration = bVal;};

protected:
	void	Pix2Board(POINTPIX pix, FovIndex index, POINT2D *xyBoard);
	void	SolveXOneIteration();
	void	ZeroTheSystem();
	void	FillMatrixA();
	void	ConstrainPerTrig(bool bPinPanelWithCalibration);
	//void	ConstrainCalDrift(); // merge into ConstrainPerTrig

private:
	unsigned int	_iCalDriftStartCol;
	//double*			_dCalDriftEst;
	double*			_dThetaEst;
	fitInfo*		_fitInfo;
	unsigned int	_iCorrelationNum;  // count into _fitInfo array

	bool	_bPinPanelWithCalibration;
};