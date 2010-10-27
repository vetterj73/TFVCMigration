#pragma once



namespace MosaicDM 
{
	//typedef double[3][3] Transform; 
	class MosaicLayer;
	class MosaicTile
	{
		public:
			MosaicTile();
			~MosaicTile(void);
			void Initialize(MosaicLayer* pMosaicLayer);	
			unsigned char *	GetImageBuffer(){return _pImageBuffer;};	
			void SetImageBuffer(unsigned char* pImageBuffer){_pImageBuffer = pImageBuffer;};
			bool ContainsImage(){return _pImageBuffer != NULL;}

		private:
			MosaicLayer *_pMosaicLayer;
			unsigned char * _pImageBuffer;
		//	Transform _nominalTransform;
		//	Transform _resultTransform;
	};
}
