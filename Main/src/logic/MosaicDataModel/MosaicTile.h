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
			void SetImageBuffer(unsigned char* pImageBuffer){_pImageBuffer = pImageBuffer;};

			///
			///	returns true is this mosaic contains an image.
			///
			bool ContainsImage(){return _pImageBuffer != NULL;}

		protected:
			///
			///	Called by the MosaicLayer class.
			///
			void Initialize(MosaicLayer* pMosaicLayer);	

		private:
			MosaicLayer *_pMosaicLayer;
			unsigned char * _pImageBuffer;
	};
}
