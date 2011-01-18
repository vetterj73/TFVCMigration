#include "StdAfx.h"
#include "MosaicTile.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"

namespace MosaicDM 
{
	MosaicTile::MosaicTile()
	{
		_pMosaicLayer = NULL;
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
	/// rotation angle in radians, offset X and y for image origin in world space
	///
	void MosaicTile::SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
		double rotation, double offsetXInMeters, double offsetYInMeters)
	{
		ImgTransform inputTransform;
		inputTransform.Config(pixelSizeXInMeters, pixelSizeYInMeters,rotation, offsetXInMeters, offsetYInMeters);
		
		Configure(
			_pMosaicLayer->GetMosaicSet()->GetImageWidthInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageHeightInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageStrideInPixels(), 
			inputTransform, inputTransform, false, NULL);
	}
}
