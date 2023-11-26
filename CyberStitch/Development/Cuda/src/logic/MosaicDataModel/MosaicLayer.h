#pragma once

#include "Image.h"

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
			MosaicTile* GetTile(unsigned int cameraIndex, unsigned int triggerIndex);

			///
			///	Get the Image Subclass at a given camera and trigger of the mosaic.
			/// Returns null if this row or column is out of range.
			///
			Image* GetImage(int cameraIndex, int triggerIndex);

			///
			///	Returns the number of tiles in this (or any?) layer.
			///
			unsigned int GetNumberOfTiles();

			///
			///	Does this layer have all of its images?
			///
			bool HasAllImages();

			unsigned int Index();

			///
			///	Gets the set this layer belongs to
			///
			MosaicSet *GetMosaicSet(){return _pMosaicSet;};

			unsigned int GetNumberOfTriggers(){return _numTriggers;};
			unsigned int GetNumberOfCameras(){return _numCameras;};

			bool IsAlignWithCad() {return _bAlignWithCAD;};
			bool IsAlignWithFiducial() {return _bAlignWithFiducial;};
			void SetAlignWithCad(bool bAlignWithCad) { _bAlignWithCAD = bAlignWithCad;};
			void SetAlignWithFiducial(bool bAlignWithFiducial) { _bAlignWithFiducial = bAlignWithFiducial;};

			///
			///	Clears all images from this layer
			///
			void ClearAllImages();

			void CameraCentersInY(double* pdCenY);
			void TriggerCentersInX(double* pdCenX);

			Image* GetMaskImage(unsigned int iCamIndex, unsigned int iTrigIndex);
			bool PrepareMaskImages();

			///
			///	Get the stitched buffer for the image... this needs to be filled in by alignment...
			///
			Image *GetStitchedImage();

			void SetStitchedBuffer(unsigned char *pBuffer);
			
		protected:
			/// Called from MosaicSet when a layer is added.
			void Initialize(MosaicSet *pMosaicSet, 
        		unsigned int numCameras,
				unsigned int numTriggers,
				bool bAlignWithCAD,
				bool bAlignWithFiducial,
				unsigned int layerIndex);


			void CreateStitchedImageIfNecessary();
			void AllocateStitchedImageIfNecessary();

			///
			///	Adds an image...
			///
			bool AddImage(unsigned char *pBuffer, unsigned int cameraIndex, unsigned int triggerIndex);

		private:
			bool _stitchedImageValid;
			unsigned int _numTriggers;
			unsigned int _numCameras;
			MosaicSet *_pMosaicSet;
			MosaicTile *_pTileArray;
			Image* _maskImages;				// An array fo mask images 
			bool _bIsMaskImgValid;			// Flag of whether mask images are valid to use
			bool _bAlignWithCAD;
			bool _bAlignWithFiducial;
			unsigned int _layerIndex;       // Not sure why this is needed but it is used in alignment..
	
			Image *_pStitchedImage;
	};
}
