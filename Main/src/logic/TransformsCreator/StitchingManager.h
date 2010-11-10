#pragma once
#include"OverlapManager.h"

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
	unsigned int TriggerIndex;
	unsigned int CameraIndex;
};

bool operator<(const FovIndex& a, const FovIndex& b);
bool operator>(const FovIndex& a, const FovIndex& b);

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
	OverlapManager* _pOvelapManager;


};

