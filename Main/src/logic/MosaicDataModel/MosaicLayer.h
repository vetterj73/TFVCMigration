#pragma once


namespace MosaicDM 
{
	class MosaicSet;
	class MosaicTile;

	///
	///	MosaicLayer is one layer of an MosaicSet.  In the case of SIM (one of the clients), a layer would be
	/// one illumination (one capture spec's worth of images).
	///
	class MosaicLayer
	{
		public:
			///
			///	Constructor
			///
			MosaicLayer();

			///
			///	Destructor
			///
			~MosaicLayer(void);

			///@todo - could be private/friend relationship... doesn't need to be public...
			void Initialize(MosaicSet *pMosaicSet, double offsetInMM);
	
			///
			///	Get the tile at a given row and column of the mosaic.
			/// Returns null if this row or column is out of range.
			///
			MosaicTile* GetTile(int row, int column);
			
			///
			///	Returns the number of tiles in this (or any?) layer.
			///
			int NumberOfTiles();

			///
			///	Does this layer have all of its images?
			///
			bool HasAllImages();

		private:
			MosaicSet *_pMosaicSet;
			MosaicTile *_pTileArray;
			double _offsetInMM;
	};
}
