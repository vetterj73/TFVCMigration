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