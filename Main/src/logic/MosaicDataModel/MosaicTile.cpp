#include "StdAfx.h"
#include "MosaicTile.h"

namespace CyberStitch 
{
	MosaicTile::MosaicTile()
	{
		_pMosaicLayer = NULL;
		_pImageBuffer = NULL;
	}

	MosaicTile::~MosaicTile(void)
	{
	}

	void MosaicTile::Initialize(MosaicLayer* pMosaicLayer)
	{
		_pMosaicLayer = pMosaicLayer;
	}
}
