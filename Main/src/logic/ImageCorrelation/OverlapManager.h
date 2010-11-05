/*
 Class to create and manage the overlaps
*/

#pragma once

#include "MosaicImage.h"
#include "CorrelationFlags.h"
#include "OverlapDefines.h"

using namespace MosaicDM;

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

protected:
	void CreateFovFovOverlaps();	
	void CreateCadFovOverlaps();
	void CreateFidFovOverlaps();
	
	bool CreateFovFovOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2);
private:	
	MosaicImage* _pMosaics;
	CorrelationFlags** _pFlags;	
	unsigned int _iNumIllumination;
	
	Image* _pCadImg;
	DRect _validRect;

	unsigned int _iNumImgX;
	unsigned int _iNumImgY;
	
	// A[Mosaic Index][Row(y) Index][Column(x) Index]
	list<FovFovOverlap>*** _fovFovOverlapLists;
	list<CadFovOverlap>*** _cadFovOverlapLists;
	list<FidFovOverlap>*** _fidFovOverlapLists;

	unsigned int _iMinOverlapSize;
};

