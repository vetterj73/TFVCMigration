#pragma once


namespace MosaicDM 
{
	class MosaicSet;
	class MosaicTile;

	//
	//	MosaicLayer is one layer of a overlapping mosaic (usually defined by a particular illumination, at least in the 
	//  case of SIM).
	//
	class MosaicLayer
	{
		public:
			MosaicLayer();
			~MosaicLayer(void);

			///@todo - could be private/friend relationship... doesn't need to be public...
			void Initialize(MosaicSet *pMosaicSet, double offsetInMM);
			MosaicTile* GetTile(int row, int column);
			
			int NumberOfTiles();
			bool HasAllImages();

		private:
			MosaicSet *_pMosaicSet;
			MosaicTile *_pTileArray;
			double _offsetInMM;
	};
}