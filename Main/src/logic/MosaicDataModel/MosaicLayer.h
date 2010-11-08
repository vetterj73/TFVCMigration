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
			int GetNumberOfTiles();

			///
			///	Does this layer have all of its images?
			///
			bool HasAllImages();

			///
			///	Gets the set this layer belongs to
			///
			MosaicSet *GetMosaicSet(){return _pMosaicSet;};

			int GetNumberOfTriggers(){return _numTriggers;}
			int GetNumberOfCameras(){return _numCameras;}
			double GetCameraOverlapInMeters(){return _cameraOverlap;}
			double GetTriggerOverlapInMeters(){return _triggerOverlap;}
			double GetCameraOffsetInMeters(){return _cameraOffset;}
			double GetTriggerOffsetInMeters(){return _triggerOffset;}

		protected:
			/// Called from MosaicSet when a layer is added.
			void Initialize(MosaicSet *pMosaicSet, 
				double cameraOffsetInMeters, 
				double triggerOffsetInMeters,
        		int numCameras,
				double cameraOverlapInMeters,
				int numTriggers,
				double triggerOverlapInMeters,
				bool correlateWithCAD);

			///
			///	Adds an image...
			///
			bool AddImage(unsigned char *pBuffer, int cameraIndex, int triggerIndex);

		private:
			int _numTriggers;
			int _numCameras;
			double _triggerOverlap;
			double _cameraOverlap;
			double _triggerOffset;
			double _cameraOffset;
			MosaicSet *_pMosaicSet;
			MosaicTile *_pTileArray;
			bool _correlateWithCAD;
	};
}
