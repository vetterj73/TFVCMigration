#include "StdAfx.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "MosaicTile.h"

namespace CyberStitch 
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
		int numTiles = pMosaicSet->GetNumRows()*pMosaicSet->GetNumColumns();
		_pTileArray = new MosaicTile[numTiles];

		for(int i=0; i<numTiles; i++)
		{
			_pTileArray[i].Initialize(this);
		}
	}

	MosaicTile* MosaicLayer::GetTile(int row, int column)
	{
		if(row<0 || row>=_pMosaicSet->GetNumRows() || column<0 || column>_pMosaicSet->GetNumColumns())
			return NULL;

		return &_pTileArray[row*_pMosaicSet->GetNumColumns()+column];
	}
}