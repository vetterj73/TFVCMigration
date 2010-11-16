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
	///	Create transform here
	/// roatation angle in radians, offset X and y for image origin in world space
	///
	void MosaicTile::SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
		double rotation,
		double offsetXInMeters, double offsetYInMeters)
	{
		_inputTransform.Config(pixelSizeXInMeters, pixelSizeYInMeters,rotation, offsetXInMeters, offsetYInMeters);
	}
}
