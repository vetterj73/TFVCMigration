#pragma once

#include "Image.h"

namespace MosaicDM 
{
	class MosaicLayer;

	///
	///	MosaicTile is one tile of a MosaicLayer.  It contains the image and its transforms
	///
	class MosaicTile
	{
		public:
			friend class MosaicLayer;

			///
			///	Constructor.  Need to call Initialize to setup the object
			///
			MosaicTile();

			///
			///	Destructor
			///
			~MosaicTile(void);

			///
			///	returns true is this mosaic contains an image.
			///
			bool ContainsImage(){return _containsImage;}

			///
			///	Called by the MosaicLayer class.
			///
			void Initialize(MosaicLayer* pMosaicLayer, double centerOffsetX, double centerOffsetY);	

			///
			///	Sets the parameters needed for transform.  If this function isn't called,
			/// nominal values will be used.
			///
			void SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
				double rotation, double centerOffsetXInMeters, double centerOffsetYInMeters);

			///
			/// Set camera calibration parameters
			///
			void	SetTransformCamCalibration(TransformCamModel t);
			void	SetTransformCamCalibrationS(unsigned int i, float val);
			void	SetTransformCamCalibrationdSdz(unsigned int i, float val);
			void	SetTransformCamCalibrationUMax(double val);
			void	SetTransformCamCalibrationVMax(double val);
			void	ResetTransformCamCalibration();
			void	ResetTransformCamModel();

			Image* GetImagPtr() {return _pImage;};

			bool Bayer2Lum(int iBayerTypeIndex);

		protected:
			void ClearImageBuffer()
			{
				_containsImage = false;
			}

			bool SetImageBuffer(unsigned char* pImageBuffer);

		private:
			MosaicLayer *_pMosaicLayer;
			bool _containsImage;
			Image* _pImage;
	};
}
