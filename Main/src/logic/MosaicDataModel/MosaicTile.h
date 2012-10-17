#pragma once

#include "Image.h"

namespace MosaicDM 
{
	class MosaicLayer;

	struct TilePosition
	{
		unsigned int iTrigIndex;
		unsigned int iCamIndex;
		
		TilePosition()
		{
			iTrigIndex = -1;
			iCamIndex = -1;
		}

		TilePosition(unsigned int iTrig, unsigned int iCam)
		{
			iTrigIndex = iTrig;
			iCamIndex = iCam;
		}
	};

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

			/// return true if the fov was added to aligner
			bool Added2Aligner() {return _bAdded2Aligner;};
			void SetAdded2Aligner() {_bAdded2Aligner = true;};

			///
			///	Called by the MosaicLayer class.
			///
			void Initialize(MosaicLayer* pMosaicLayer, double centerOffsetX, double centerOffsetY);	

			///
			///	Sets the parameters needed for nominal transform.  If this function isn't called,
			/// nominal values will be used.
			///
			void SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
				double rotation, double centerOffsetXInMeters, double centerOffsetYInMeters);
			
			/// Set nominal transform pTrans is a size 8 arrary for projecitve transform
			void SetNominalTransform(double dTrans[9]);

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
				_bAdded2Aligner = false;
			}

			bool SetRawImageBuffer(unsigned char* pImageBuffer);

			bool SetYCrCbImageBuffer(unsigned char* pImageBuffer);

		private:
			MosaicLayer *_pMosaicLayer;
			bool _containsImage;
			Image* _pImage;
			bool _bAdded2Aligner;
	};
}
