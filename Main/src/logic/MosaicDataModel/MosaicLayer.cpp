#include "StdAfx.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "MosaicTile.h"

namespace MosaicDM 
{
	MosaicLayer::MosaicLayer()
	{
		_pMosaicSet = NULL;
		_pTileArray = NULL;
		_cameraOffset = 0;
		_triggerOffset = 0;
		_numCameras = 0;
		_numTriggers = 0;
		_cameraOverlap = 0;
		_triggerOverlap = 0;
		_correlateWithCAD = false;
	}

	MosaicLayer::~MosaicLayer(void)
	{
		delete[] _pTileArray;
	}

	void MosaicLayer::Initialize(MosaicSet *pMosaicSet, 
									double cameraOffsetInMeters, 
									double triggerOffsetInMeters,
        							int numCameras,
									double cameraOverlapInMeters,
									int numTriggers,
									double triggerOverlapInMeters,
									bool correlateWithCAD)
	{
		_pMosaicSet = pMosaicSet;
		_cameraOffset = cameraOffsetInMeters;
		_triggerOffset = triggerOffsetInMeters;
		_numCameras = numCameras;
		_numTriggers = numTriggers;
		_cameraOverlap = cameraOverlapInMeters;
		_triggerOverlap = triggerOverlapInMeters;
		_correlateWithCAD = correlateWithCAD;

		int numTiles = GetNumberOfTiles();
		_pTileArray = new MosaicTile[numTiles];


		for(int i=0; i<numTiles; i++)
		{
			// @todo - set X/Y center offsets nominally...
			_pTileArray[i].Initialize(this, 0.0, 0.0);
		}
	}

	MosaicTile* MosaicLayer::GetTile(int cameraIndex, int triggerIndex)
	{
		if(cameraIndex<0 || cameraIndex>=GetNumberOfCameras() || triggerIndex<0 || triggerIndex>=GetNumberOfTriggers())
			return NULL;

		return &_pTileArray[cameraIndex*GetNumberOfTriggers()+triggerIndex];
	}

	int MosaicLayer::GetNumberOfTiles()
	{
		return GetNumberOfTriggers()*GetNumberOfCameras();
	}
	
	bool MosaicLayer::HasAllImages()
	{
		int numTiles = GetNumberOfTiles();
		for(int i=0; i<numTiles; i++)
			if(!_pTileArray[i].ContainsImage())
				return false;

		return true;
	}

	bool MosaicLayer::AddImage(unsigned char *pBuffer, int cameraIndex, int triggerIndex)
	{
		MosaicTile* pTile = GetTile(cameraIndex, triggerIndex);
		if(pTile == NULL)
			return false;

		return pTile->SetImageBuffer(pBuffer);
	}
}