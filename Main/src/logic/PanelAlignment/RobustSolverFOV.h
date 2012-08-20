/*
	inherits from RobustSolver.h
*/

#pragma once


#include "RobustSolver.h"


class RobustSolverFOV: public RobustSolver
{
public:
	RobustSolverFOV(
		map<FovIndex, unsigned int>* pFovOrderMap, 
		unsigned int iMaxNumCorrelations,  
		MosaicSet* pSet, 
		bool bProjectiveTrans);

	~RobustSolverFOV(void);

	bool AddCalibationConstraints(
		MosaicLayer* pLayer, 
		unsigned int iCamIndex, 
		unsigned int iTrigIndex, 
		bool bPinFov=false, 
		bool bUseNorminalTransform=true);
	bool AddPanelEdgeContraints(
		MosaicLayer* pLayer, unsigned int iCamIndex, unsigned int iTrigIndex, 
		double dXOffset, double dSlope, bool bSlopeOnly=false);
	bool AddFovFovOvelapResults(FovFovOverlap* pOverlap);
	bool AddCadFovOvelapResults(CadFovOverlap* pOverlap);
	bool AddFidFovOvelapResults(FidFovOverlap* pOverlap);
	bool AddInputFidLocations(FiducialLocation* pLoc);

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
	
	MosaicSet*		_pSet;

private:
	bool MatchProjeciveTransform(const double pPara[12], 
		unsigned int iLayerIndex,
		unsigned int iTriggerIndex,
		unsigned int iCameraIndex, 
		double dTrans[3][3]) const;

};