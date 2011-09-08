#include "StdAfx.h"
#include "MosaicTile.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "ColorImage.h"

namespace MosaicDM 
{
	MosaicTile::MosaicTile()
	{
		_pImage = NULL;
		_pMosaicLayer = NULL;
		_containsImage = false;
	}

	MosaicTile::~MosaicTile(void)
	{
		if(_pImage != NULL) delete _pImage;
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
		inputTransform.Config(pixelSizeXInMeters, pixelSizeYInMeters, rotation, offsetXInMeters, offsetYInMeters);

		if(_pMosaicLayer->GetMosaicSet()->IsBayerPattern())
			_pImage = new ColorImage(YCrCb, true); // YCrCb color, seperate channel
			//_pImage = new ColorImage(RGB, false); // RGB color, combined channel
		else
			_pImage = new Image();
		
		_pImage->Configure(
			_pMosaicLayer->GetMosaicSet()->GetImageWidthInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageHeightInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageStrideInPixels(), 
			inputTransform, inputTransform, _pMosaicLayer->GetMosaicSet()->GetOwnBuffers(), NULL);
	}

	bool MosaicTile::SetImageBuffer(unsigned char* pImageBuffer)
	{
		if(_pMosaicLayer->GetMosaicSet()->IsBayerPattern()) // For bayer/color image
		{
			((ColorImage*)_pImage)->DemosiacFrom(pImageBuffer, 
				_pImage->Columns(), _pImage->Rows(), _pImage->Columns(),
				(BayerType)_pMosaicLayer->GetMosaicSet()->GetBayerType());

			/*/ For debug
			((ColorImage*)_pImage)->SetColorStyle(RGB);
			((ColorImage*)_pImage)->SetChannelStoreSeperated(false);
			_pImage->Save("C:\\Temp\\ColorFov.bmp"); 
			//*/
		}
		else // For gray image 
		{
			if(_pMosaicLayer->GetMosaicSet()->GetOwnBuffers())
			{
				memcpy(_pImage->GetBuffer(), pImageBuffer, _pImage->BufferSizeInBytes());	
			} 
			else
			{
				_pImage->SetBuffer(pImageBuffer);
			}
		}

		_containsImage = true;
		return true;
	}

}
