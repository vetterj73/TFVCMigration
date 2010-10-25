#pragma once



namespace CyberStitch 
{
	//typedef double[3][3] Transform; 
	class MosaicLayer;
	class MosaicTile
	{
		public:
			MosaicTile();
			~MosaicTile(void);
			void Initialize(MosaicLayer* pMosaicLayer);

		private:
			MosaicLayer *_pMosaicLayer;
		//	Transform _nominalTransform;
		//	Transform _resultTransform;
	};
}
