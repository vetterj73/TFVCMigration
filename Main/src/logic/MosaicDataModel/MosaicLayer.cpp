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
		_offsetInMeters = 0;
	}

	MosaicLayer::~MosaicLayer(void)
	{
		delete[] _pTileArray;
	}

	void MosaicLayer::Initialize(MosaicSet *pMosaicSet, double offsetInMeters)
	{
		_pMosaicSet = pMosaicSet;
		_offsetInMeters = offsetInMeters;
		int numTiles = NumberOfTiles();
		_pTileArray = new MosaicTile[numTiles];


		for(int i=0; i<numTiles; i++)
		{
			// @todo - set X/Y center offsets nominally...
			_pTileArray[i].Initialize(this, 0.0, 0.0);
		}
	}

	MosaicTile* MosaicLayer::GetTile(int cameraIndex, int triggerIndex)
	{
		if(cameraIndex<0 || cameraIndex>=_pMosaicSet->GetNumCameras() || triggerIndex<0 || triggerIndex>=_pMosaicSet->GetNumTriggers())
			return NULL;

		return &_pTileArray[cameraIndex*_pMosaicSet->GetNumTriggers()+triggerIndex];
	}

	int MosaicLayer::NumberOfTiles()
	{
		return _pMosaicSet->GetNumTriggers()*_pMosaicSet->GetNumCameras();
	}
	
	bool MosaicLayer::HasAllImages()
	{
		int numTiles = NumberOfTiles();
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