#pragma once
#include "MosaicTile.h"
#include "ImgTransformCamModel.h"

namespace MMosaicDM 
{
	///
	///	Simple Wrapper around unmanaged MosaicTile.  Only exposes what is necessary.
	///
	public ref class ManagedMosaicTile
	{
		public:
			///
			///	Constructor
			///
			ManagedMosaicTile(MosaicDM::MosaicTile * pMosaicTile)
			{
				_pMosaicTile = pMosaicTile;
			}
		
			///
			///	GetImageBuffer - returns the image buffer.
			///
			System::IntPtr GetImageBuffer()
			{
				return (System::IntPtr)_pMosaicTile->GetImagPtr()->GetBuffer();
			}

			bool Bayer2Lum(int iType)
			{
				return _pMosaicTile->Bayer2Lum(iType);
			}

			///
			///	Sets the parameters needed for transform.  If this function isn't called,
			/// nominal values will be used.
			///
			void SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
				double rotation,
				double centerOffsetXInMeters, double centerOffsetYInMeters)
			{
				_pMosaicTile->SetTransformParameters(pixelSizeXInMeters, pixelSizeYInMeters,
					rotation,
					centerOffsetXInMeters, centerOffsetYInMeters);
			}
			///
			/// Set camera calibration parameters
			///
			//void	SetTransformCamCalibration(TransformCamModel t);
			void	SetTransformCamCalibrationS(unsigned int i, float val)
			{
				_pMosaicTile->SetTransformCamCalibrationS(i, val);
			}
			void	SetTransformCamCalibrationdSdz(unsigned int i, float val)
			{
				_pMosaicTile->SetTransformCamCalibrationdSdz(i, val);
			}
			void	SetTransformCamCalibrationUMax(double val)
			{
				_pMosaicTile->SetTransformCamCalibrationUMax(val);
			}
			void	SetTransformCamCalibrationVMax(double val)
			{
				_pMosaicTile->SetTransformCamCalibrationVMax(val);
			}
			void	ResetTransformCamCalibration()
			{
				_pMosaicTile->ResetTransformCamCalibration();
			}
			void	ResetTransformCamModel()
			{
				_pMosaicTile->ResetTransformCamModel();
			}

		private:
			MosaicDM::MosaicTile *_pMosaicTile;
	};
}

