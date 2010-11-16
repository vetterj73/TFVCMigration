#include "StdAfx.h"
#include "MosaicTile.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"

namespace MosaicDM 
{
	MosaicTile::MosaicTile()
	{
		_pMosaicLayer = NULL;
		_pImageBuffer = NULL;
	}

	MosaicTile::~MosaicTile(void)
	{
	}

	void MosaicTile::Initialize(MosaicLayer* pMosaicLayer, double centerOffsetX, double centerOffsetY)
	{
		_pMosaicLayer = pMosaicLayer;
	}

	///
	///	George - it makes sense to me to create an actual transform here...  
	///	Let me know your thoughts.
	///
	void MosaicTile::SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
		double rotation,
		double offsetXInMeters, double offsetYInMeters)
	{
		_inputTransform.Config(pixelSizeXInMeters, pixelSizeYInMeters,rotation, offsetXInMeters, offsetYInMeters);
	}
}
