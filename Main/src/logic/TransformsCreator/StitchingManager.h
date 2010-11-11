#pragma once
#include "OverlapManager.h"
#include "RobustSolver.h"

class StitchingManager
{
public:
	StitchingManager(OverlapManager* pOvelapManager, Image* pPanelMaskImage);
	~StitchingManager(void);

	bool AddOneImageBuffer(	
		unsigned char* pcBuf,
		unsigned int iIllumIndex, 
		unsigned int iTrigIndex, 
		unsigned int iCamIndex);	
	
	void Reset();
protected:

	bool CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap);
	bool CreateImageOrderInSolver(
		unsigned int* piIllumIndices, 
		unsigned iNumIllums, 
		map<FovIndex, unsigned int>* pOrderMap);

	bool IsReadyToCreateMasks();
	bool IsReadyToCreateTransforms();

	bool CreateMasks();
	bool CreateTransforms();
	void AddOverlapResultsForIllum(RobustSolver* solver, unsigned int iIllumIndex);

private:
	OverlapManager* _pOverlapManager;
	Image* _pPanelMaskImage;

	int _iMaskCreationStage;

	RobustSolver* _pSolver;
	map<FovIndex, unsigned int> _solverMap;

	RobustSolver* _pMaskSolver;
	map<FovIndex, unsigned int> _maskMap;
	bool _bMasksCreated;
};

