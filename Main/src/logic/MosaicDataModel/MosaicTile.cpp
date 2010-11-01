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
		_rotation = 0;
		_pixelSizeX = _pMosaicLayer->GetMosaicSet()->GetNominalPixelSizeX();
		_pixelSizeY = _pMosaicLayer->GetMosaicSet()->GetNominalPixelSizeY();
		_centerOffsetX = centerOffsetX;
		_centerOffsetY = centerOffsetY;
	}

	///
	///	George - it makes sense to me to create an actual transform here...  
	///	Let me know your thoughts.
	///
	void MosaicTile::SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
		double centerOffsetXInMeters, double centerOffsetYInMeters,
		double rotation)
	{
		_pixelSizeX=pixelSizeXInMeters;
		_pixelSizeY=pixelSizeYInMeters;
		_centerOffsetX=centerOffsetXInMeters;
		_centerOffsetY=centerOffsetYInMeters;
		_rotation=rotation;
	}
}
