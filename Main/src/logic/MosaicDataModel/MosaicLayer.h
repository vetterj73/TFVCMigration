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
			friend class MosaicSet;

			///
			///	Constructor
			///
			MosaicLayer();

			///
			///	Destructor
			///
			~MosaicLayer(void);

			///
			///	Get the tile at a given camera and trigger of the mosaic.
			/// Returns null if this row or column is out of range.
			///
			MosaicTile* GetTile(int cameraIndex, int triggerIndex);
			
			///
			///	Returns the number of tiles in this (or any?) layer.
			///
			int NumberOfTiles();

			///
			///	Does this layer have all of its images?
			///
			bool HasAllImages();

		protected:
			/// Called from MosaicSet when a layer is added.
			void Initialize(MosaicSet *pMosaicSet, double offsetInMeters);

			///
			///	Adds an image...
			///
			bool AddImage(unsigned char *pBuffer, int cameraIndex, int triggerIndex);

		private:
			MosaicSet *_pMosaicSet;
			MosaicTile *_pTileArray;
			double _offsetInMeters;
	};
}
