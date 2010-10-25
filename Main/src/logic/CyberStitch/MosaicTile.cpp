#include "StdAfx.h"
#include "MosaicTile.h"

namespace CyberStitch 
{
	MosaicTile::MosaicTile()
	{
	}

	MosaicTile::~MosaicTile(void)
	{
	}

	void MosaicTile::Initialize(MosaicLayer* pMosaicLayer)
	{
		_pMosaicLayer = pMosaicLayer;
	}
}
