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
			///	Called by the MosaicLayer class.
			///
			void Initialize(MosaicLayer* pMosaicLayer, double centerOffsetX, double centerOffsetY);	

			///
			///	Sets the parameters needed for transform.  If this function isn't called,
			/// nominal values will be used.
			///
			void SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
				double rotation,
				double centerOffsetXInMeters, double centerOffsetYInMeters);

		protected:
			bool SetImageBuffer(unsigned char* pImageBuffer)
			{
				if(_pImageBuffer != NULL)
					return false;

				_pImageBuffer = pImageBuffer;
				return true;
			};

		private:
			MosaicLayer *_pMosaicLayer;
			unsigned char * _pImageBuffer;
			ImgTransform _inputTransform;
			ImgTransform _outputTransform;
	};
}
