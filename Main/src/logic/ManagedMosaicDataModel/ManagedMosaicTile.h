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
			///	Sets the parameters needed for nominal transform.  If this function isn't called,
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

			void SetNominalTransform(array<double>^ pTrans)
			{
				double dTrans[9];
				for(int i=0; i<8; i++)
					dTrans[i] = pTrans[i];
				dTrans[8] = 1;

				_pMosaicTile->SetNominalTransform(dTrans);
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

			bool SetCamModelLinearCalib(array<double>^ pdVal)
			{
				if(pdVal->Length != 8)
					return(false);

				double trans[8];
				for(int i=0; i<8; i++)
					trans[i] = pdVal[i];

				_pMosaicTile->SetCamModelLinearCalib(trans);

				return(true);
			}

		private:
			MosaicDM::MosaicTile *_pMosaicTile;
	};
}

