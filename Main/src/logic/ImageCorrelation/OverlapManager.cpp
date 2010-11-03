#include "OverlapManager.h"

OverlapManager::OverlapManager(MosaicImage* pMosaics, CorrelationFlags** pFlags, unsigned int iNumIllumination)
{
	_pMosaics = pMosaics;	
	_pFlags = pFlags;
	_iNumIllumination = iNumIllumination;

	unsigned int i, j;
	_iSizeX=0;
	_iSizeY=0;
	for(i=0; i<_iNumIllumination; i++)
	{
		if (_iSizeX < pMosaics[i].NumImInX())
			_iSizeX = pMosaics[i].NumImInX();

		if (_iSizeY < pMosaics[i].NumImInY())
			_iSizeY = pMosaics[i].NumImInY();
	}

	// Create 3D arrays for storage of overlaps
	_fovFovOverlapLists = new list<FovFovOverlap>**[_iNumIllumination];
	_cadFovOverlapLists = new list<CadFovOverlap>**[_iNumIllumination];
	_fidFovOverlapLists = new list<FidFovOverlap>**[_iNumIllumination];
	for(i=0; i<_iNumIllumination; i++)
	{
		_fovFovOverlapLists[i] = new list<FovFovOverlap>*[_iSizeY];
		_cadFovOverlapLists[i] = new list<CadFovOverlap>*[_iSizeY];
		_fidFovOverlapLists[i] = new list<FidFovOverlap>*[_iSizeY];

		for(j=0; j<_iSizeY; j++)
		{
			_fovFovOverlapLists[i][j] = new list<FovFovOverlap>[_iSizeX];
			_cadFovOverlapLists[i][j] = new list<CadFovOverlap>[_iSizeX];
			_fidFovOverlapLists[i][j] = new list<FidFovOverlap>[_iSizeX];
		}
	}
}


OverlapManager::~OverlapManager(void)
{
	// Release 3D arrays for storage
	unsigned int i, j;
	for(i=0; i<_iNumIllumination; i++)
	{
		for(j=0; j<_iSizeY; j++)
		{
			delete [] _fovFovOverlapLists[i][j];
			delete [] _cadFovOverlapLists[i][j];
			delete [] _fidFovOverlapLists[i][j];
		}

		delete [] _fovFovOverlapLists[i];
		delete [] _cadFovOverlapLists[i];
		delete [] _fidFovOverlapLists[i];
	}
	delete [] _fovFovOverlapLists;
	delete [] _cadFovOverlapLists;
	delete [] _fidFovOverlapLists;
}


bool OverlapManager::CreateOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2)
{
	CorrelationFlags flags = _pFlags[iIndex1][iIndex2];
	bool bColCol = flags.GetCameraToCamera();
	bool bRowRow = flags.GetTriggerToTrigger();
	bool bMask = flags.GetMaskNeeded();
	bool bCad	= false; //need modify

	return true;
}
