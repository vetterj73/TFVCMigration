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
	}

	MosaicLayer::~MosaicLayer(void)
	{
		delete[] _pTileArray;
	}

	void MosaicLayer::Initialize(MosaicSet *pMosaicSet, double offsetInMM)
	{
		_pMosaicSet = pMosaicSet;
		_offsetInMM = offsetInMM;
		int numTiles = NumberOfTiles();
		_pTileArray = new MosaicTile[numTiles];

		for(int i=0; i<numTiles; i++)
		{
			_pTileArray[i].Initialize(this);
		}
	}

	MosaicTile* MosaicLayer::GetTile(int trigger, int camera)
	{
		if(trigger<0 || trigger>=_pMosaicSet->GetNumTriggers() || camera<0 || camera>_pMosaicSet->GetNumCameras())
			return NULL;

		return &_pTileArray[trigger*_pMosaicSet->GetNumCameras()+camera];
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
}