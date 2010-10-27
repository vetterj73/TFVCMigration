#pragma once

namespace MosaicDM 
{
	class MosaicLayer;

	///
	///	MosaicTile is one tile of a MosaicLayer.  It contains the image and its transforms
	///
	class MosaicTile
	{
		public:
			///
			///	Constructor.  Need to call Initialize to setup the object
			///
			MosaicTile();

			///
			///	Destructor
			///
			~MosaicTile(void);

			///
			///	Called by the MosaicLayer class (need not be called from the outside).
			///
			void Initialize(MosaicLayer* pMosaicLayer);	

			///
			///	Gets/Sets the image buffer (assumed to be the size defined by the MosaicSet
			///
			unsigned char *	GetImageBuffer(){return _pImageBuffer;};	
			void SetImageBuffer(unsigned char* pImageBuffer){_pImageBuffer = pImageBuffer;};

			///
			///	returns true is this mosaic contains an image.
			///
			bool ContainsImage(){return _pImageBuffer != NULL;}

		private:
			MosaicLayer *_pMosaicLayer;
			unsigned char * _pImageBuffer;
	};
}
