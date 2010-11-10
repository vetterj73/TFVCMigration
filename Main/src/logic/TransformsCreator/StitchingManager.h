#pragma once
#include "OverlapManager.h"
#include "RobustSolver.h"

class StitchingManager
{
public:
	StitchingManager(OverlapManager* pOvelapManager);
	~StitchingManager(void);

	bool AddOneImageBuffer(unsigned int iIllumIndex, unsigned int iTrigIndex, unsigned int iCamIndex);

protected:
	void reset();
	bool CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap);
	bool CreateImageOrderInSolver(
		unsigned int* piIllumIndices, 
		unsigned iNumIllums, 
		map<FovIndex, unsigned int>* pOrderMap);

private:
	OverlapManager* _pOverlapManager;

	int _iMaskCreationStage;

	RobustSolver* _pSolver;
	map<FovIndex, unsigned int> _solverMap;

	RobustSolver* _pMaskSolver;
	map<FovIndex, unsigned int> _maskMap;
};

