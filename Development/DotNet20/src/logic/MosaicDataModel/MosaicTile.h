#pragma once

#include "Image.h"

namespace MosaicDM 
{
	class MosaicLayer;

	///
	///	MosaicTile is one tile of a MosaicLayer.  It contains the image and its transforms
	///
	class MosaicTile : public Image
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
			bool ContainsImage(){return GetBuffer() != NULL;}

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

		protected:
			void ClearImageBuffer()
			{
				SetBuffer(NULL);
			}

			bool SetImageBuffer(unsigned char* pImageBuffer)
			{
				if(GetBuffer() != NULL)
					return false;

				SetBuffer(pImageBuffer);
				return true;
			};

		private:
			MosaicLayer *_pMosaicLayer;
	};
}
