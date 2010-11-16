#pragma once
#include "OverlapManager.h"
#include "RobustSolver.h"

class StitchingManager
{
public:
	StitchingManager(OverlapManager* pOverlapManager, Image* pPanelMaskImage);
	~StitchingManager(void);

	bool AddOneImageBuffer(	
		unsigned char* pcBuf,
		unsigned int iIllumIndex, 
		unsigned int iTrigIndex, 
		unsigned int iCamIndex);	
	
	void Reset();

	void CreateStitchingImage(unsigned int iIllumIndex, Image* pPanelImage) const;
	static void CreateStitchingImage(const MosaicImage* pMosaic, Image* pPanelImage);

protected:
	bool CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap) const;
	bool CreateImageOrderInSolver(
		unsigned int* piIllumIndices, 
		unsigned iNumIllums, 
		map<FovIndex, unsigned int>* pOrderMap) const;

	bool IsReadyToCreateMasks() const;
	bool IsReadyToCreateTransforms() const;

	bool CreateMasks();
	bool CreateTransforms();
	void AddOverlapResultsForIllum(RobustSolver* solver, unsigned int iIllumIndex);

	void StitchingManager::SaveStitchingImages(string name, unsigned int iNum);

private:
	OverlapManager* _pOverlapManager;
	Image* _pPanelMaskImage;

	int _iMaskCreationStage;

	RobustSolver* _pSolver;
	map<FovIndex, unsigned int> _solverMap;

	RobustSolver* _pMaskSolver;
	map<FovIndex, unsigned int> _maskMap;
	bool _bMasksCreated;

	// For debug;
	bool _bDebug;
};

