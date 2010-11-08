/*
 Class to create and manage the overlaps
*/

#pragma once

#include "MosaicImage.h"
#include "CorrelationFlags.h"
#include "OverlapDefines.h"

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
	unsigned int TriggerIndex;
	unsigned int CameraIndex;
};




class OverlapManager
{
public:
	OverlapManager(
		MosaicImage* pMosaics, 
		CorrelationFlags** pFlags, 
		unsigned int iNumIllumination,
		Image* pCadImg, 
		DRect validRect);
	~OverlapManager(void);

	bool ResetforNewPanel();

	bool DoAlignmentForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex);

	list<FovFovOverlap>* GetFovFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	list<CadFovOverlap>* GetCadFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	list<FidFovOverlap>* GetFidFovListForFov(
		unsigned int iMosaicIndex, 
		unsigned int iTrigIndex,
		unsigned int iCamIndex) const;

	bool CreateImageOrderInSolver(
		unsigned int* piIllumIndices, 
		unsigned iNumIllums, 
		map<FovIndex, unsigned int>* pOrderMap);

	bool CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap);

protected:
	void CreateFovFovOverlaps();	
	void CreateCadFovOverlaps();
	void CreateFidFovOverlaps();
	
	bool CreateFovFovOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2);
	
	void MaskCreationStage();

private:	
	MosaicImage* _pMosaics;
	CorrelationFlags** _pFlags;	
	unsigned int _iNumIllumination;
	
	Image* _pCadImg;
	DRect _validRect;

	unsigned int _iNumCameras;
	unsigned int _iNumTriggers;
	
	// A[Mosaic Index][Row(y) Index][Column(x) Index]
	list<FovFovOverlap>*** _fovFovOverlapLists;
	list<CadFovOverlap>*** _cadFovOverlapLists;
	list<FidFovOverlap>*** _fidFovOverlapLists;

	unsigned int _iMinOverlapSize;

	int _iMaskCreationStage;
};

