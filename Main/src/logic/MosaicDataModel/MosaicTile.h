#pragma once

#include "ImgTransform.h"

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
			///	Gets/Sets the image buffer (assumed to be the size defined by the MosaicSet
			///
			unsigned char *	GetImageBuffer(){return _pImageBuffer;};	

			///
			///	returns true is this mosaic contains an image.
			///
			bool ContainsImage(){return _pImageBuffer != NULL;}

			///
			///	Sets the parameters needed for transform.  If this function isn't called,
			/// nominal values will be used.
			///
			void SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
				double centerOffsetXInMeters, double centerOffsetYInMeters,
				double rotation);

			double GetPixelSizeX(){return _pixelSizeX;};
			double GetPixelSizeY(){return _pixelSizeY;};
			double CenterOffsetX(){return _centerOffsetX;};
			double CenterOffsetY(){return _centerOffsetY;};
			double Rotation(){return _rotation;};

		protected:
			bool SetImageBuffer(unsigned char* pImageBuffer)
			{
				if(_pImageBuffer != NULL)
					return false;

				_pImageBuffer = pImageBuffer;
				return true;
			};

			///
			///	Called by the MosaicLayer class.
			///
			void Initialize(MosaicLayer* pMosaicLayer, double centerOffsetX, double centerOffsetY);	

		private:
			MosaicLayer *_pMosaicLayer;
			unsigned char * _pImageBuffer;
			double _pixelSizeX;
			double _pixelSizeY;
			double _centerOffsetX;
			double _centerOffsetY;
			double _rotation;
			ImgTransform _inputTransform;
			ImgTransform _outputTransform;
	};
}
