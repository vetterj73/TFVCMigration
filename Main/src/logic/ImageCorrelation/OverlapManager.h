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
	OverlapManager(MosaicImage* pMosiacs, CorrelationFlags** pFlags , unsigned int iNumIllumination);
	~OverlapManager(void);

protected:
	bool CreateOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2);

private:	
	MosaicImage* _pMosaics;
	CorrelationFlags** _pFlags;
	unsigned int _iNumIllumination;
	unsigned int _iSizeX;
	unsigned int _iSizeY;
	
	list<FovFovOverlap>*** _fovFovOverlapLists;
	list<CadFovOverlap>*** _cadFovOverlapLists;
	list<FidFovOverlap>*** _fidFovOverlapLists;
};

