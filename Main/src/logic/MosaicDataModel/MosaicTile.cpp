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
		_bAdded2Aligner = false;
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

		if(_pMosaicLayer->GetMosaicSet()->IsBayerPattern() && !_pMosaicLayer->GetMosaicSet()->IsSkipDemosaic())
			_pImage = new ColorImage(YCrCb, true); // YCrCb color, seperate channel
			//_pImage = new ColorImage(BGR, false); // RGB color, combined channel
		else
			_pImage = new Image();
		
		_pImage->Configure(
			_pMosaicLayer->GetMosaicSet()->GetImageWidthInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageHeightInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageStrideInPixels(), 
			inputTransform, inputTransform, _pMosaicLayer->GetMosaicSet()->HasOwnBuffers(), NULL);

	}


	/// Set nominal transform pTrans is a size 8 arrary for projecitve transform
	void MosaicTile::SetNominalTransform(double dTrans[9])
	{
		ImgTransform inputTransform;
		inputTransform.SetMatrix(dTrans);

		if(_pMosaicLayer->GetMosaicSet()->IsBayerPattern() && !_pMosaicLayer->GetMosaicSet()->IsSkipDemosaic())
			_pImage = new ColorImage(YCrCb, true); // YCrCb color, seperate channel
			//_pImage = new ColorImage(BGR, false); // RGB color, combined channel
		else
			_pImage = new Image();
		
		_pImage->Configure(
			_pMosaicLayer->GetMosaicSet()->GetImageWidthInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageHeightInPixels(), 
			_pMosaicLayer->GetMosaicSet()->GetImageStrideInPixels(), 
			inputTransform, inputTransform, _pMosaicLayer->GetMosaicSet()->HasOwnBuffers(), NULL);
	}

	///
	/// TODO 
	/// Set camera calibration parameters
	///
	void MosaicTile::SetTransformCamCalibration(TransformCamModel t)
	{
		_pImage->SetTransformCamCalibration(t);
	}

	void	MosaicTile::SetTransformCamCalibrationS(unsigned int i, float val)
	{
		_pImage->SetTransformCamCalibrationS(i, val);
	}
	void	MosaicTile::SetTransformCamCalibrationdSdz(unsigned int i, float val)
	{
		_pImage->SetTransformCamCalibrationdSdz(i, val);
	}
	void	MosaicTile::SetTransformCamCalibrationUMax(double val)
	{
		_pImage->SetTransformCamCalibrationUMax(val);
	}
	void	MosaicTile::SetTransformCamCalibrationVMax(double val)
	{
		_pImage->SetTransformCamCalibrationVMax(val);
	}
	void	MosaicTile::ResetTransformCamCalibration()
	{
		_pImage->ResetTransformCamCalibration();
	}
	void	MosaicTile::ResetTransformCamModel()
	{
		_pImage->ResetTransformCamModel();
	}



	bool MosaicTile::SetRawImageBuffer(unsigned char* pImageBuffer)
	{
		if(_pMosaicLayer->GetMosaicSet()->IsBayerPattern() && !_pMosaicLayer->GetMosaicSet()->IsSkipDemosaic()) // For bayer/color image
		{
			// for Bayer pattern input, mosaicSet must have its own buffer
			_pMosaicLayer->GetMosaicSet()->SetOwnBuffers(true);

			if(!_pImage->HasOwnBuffer())
			{
				_pImage->CreateOwnBuffer();
			}	

			((ColorImage*)_pImage)->DemosiacFrom(pImageBuffer, 
				_pImage->Columns(), _pImage->Rows(), _pImage->Columns(),
				(BayerType)_pMosaicLayer->GetMosaicSet()->GetBayerType());

			/*/ for debug
			ImgTransform trans;
			Image tempImage(
				_pImage->Columns(), 
				_pImage->Rows(), 
				_pImage->PixelRowStride(),
				1,
				trans,
				trans,
				false,
				pImageBuffer);

			tempImage.Bayer2Lum((BayerType)_pMosaicLayer->GetMosaicSet()->GetBayerType());
			memcpy(_pImage->GetBuffer(), tempImage.GetBuffer(), _pImage->BufferSizeInBytes()/3);	
			//*/

			/*/ For debug
			((ColorImage*)_pImage)->SetColorStyle(RGB);
			((ColorImage*)_pImage)->SetChannelStoreSeperated(false);
			_pImage->Save("C:\\Temp\\ColorFov.bmp"); 
			//*/
		}
		else // For gray image 
		{
			if(_pMosaicLayer->GetMosaicSet()->HasOwnBuffers())
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

	// Input buffer need to be YCrCb
	bool MosaicTile::SetYCrCbImageBuffer(unsigned char* pImageBuffer)
	{
		if(_pMosaicLayer->GetMosaicSet()->HasOwnBuffers())
		{
			memcpy(_pImage->GetBuffer(), pImageBuffer, _pImage->BufferSizeInBytes());	
		} 
		else
		{
			_pImage->SetBuffer(pImageBuffer);
		}

		_containsImage = true;
		return true;
	}

	bool MosaicTile::Bayer2Lum(int iBayerTypeIndex)
	{
		// Validation check
		if(!_containsImage)
			return(false);

		// Bayer to luminance conversion
		return(_pImage->Bayer2Lum((BayerType)iBayerTypeIndex));
	}
}
