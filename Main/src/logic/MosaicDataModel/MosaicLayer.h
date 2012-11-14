#pragma once

#include "Image.h"
#include <map>
#include <list>
using std::map;
using std::list;

namespace MosaicDM 
{
	class SubDevicCams;
	class MosaicSet;
	class MosaicTile;

	enum FOVLRPOS
	{
		NOPREFERLR,
		LEFTFOV,
		RIGHTFOV,
	};

	enum FOVTBPOS
	{
		NOPREFERTB,
		TOPFOV,
		BOTTOMFOV
	};

	struct FOVPreferSelected
	{
		FOVLRPOS preferLR;
		FOVTBPOS preferTB;
		FOVLRPOS selectedLR;
		FOVTBPOS selectedTB;
		bool bAdjustForHeight; 
			
		FOVPreferSelected()
		{
			preferLR = NOPREFERLR;
			preferTB = NOPREFERTB;
			selectedLR = NOPREFERLR;
			selectedTB = NOPREFERTB;

			bAdjustForHeight = true; // Adjust for component height if height information is available
		}
	};

	struct ComponentHeightInfo
	{
	public:
		unsigned char* pHeightBuf;		// Component height image buf
		unsigned int iHeightSpan;		// Component height image span
		double dHeightResolution;		// Height resolution in grey level (meter/grey level)
		double dPupilDistance;			// SIM pupil distance (meter))
	};

	///
	///	MosaicLayer is one layer of an MosaicSet.  In the case of SIM (one of the clients), a layer would be
	/// one Layer (one capture spec's worth of images).
	///
	class MosaicLayer
	{
		public:
			friend class MosaicSet;
			friend class DemosaicJob;

			MosaicLayer();
			~MosaicLayer(void);

			///
			///	Get the tile at a given camera and trigger of the mosaic.
			/// Returns null if this row or column is out of range.
			///
			MosaicTile* GetTile(unsigned int triggerIndex, unsigned int cameraIndex);

			///
			///	Get the Image Subclass at a given camera and trigger of the mosaic.
			/// Returns null if this row or column is out of range.
			///
			Image* GetImage(int triggerIndex, int cameraIndex);

			///
			///	Returns the number of tiles in this (or any?) layer.
			///
			unsigned int GetNumberOfTiles();

			///
			///	Does this layer have all of its images?
			///
			bool HasAllImages();

			unsigned int Index() {return _layerIndex;};
			unsigned int DeviceIndex() 	{return _deviceIndex;};

			///
			///	Gets the set this layer belongs to
			///
			MosaicSet *GetMosaicSet(){return _pMosaicSet;};

			unsigned int GetNumberOfTriggers(){return _numTriggers;};
			unsigned int GetNumberOfCameras(){return _numCameras;};

			bool IsAlignWithCad() {return _bAlignWithCAD;};
			bool IsAlignWithFiducial() {return _bAlignWithFiducial;};
			bool IsFiducialBrighterThanBackground() {return _bFiducialBrighterThanBackground;};
			bool IsFiducialAllowNegativeMatch() {return _bFiducialAllowNegativeMatch;};
			void SetAlignWithCad(bool bAlignWithCad) { _bAlignWithCAD = bAlignWithCad;};
			void SetAlignWithFiducial(bool bAlignWithFiducial) { _bAlignWithFiducial = bAlignWithFiducial;};

			// After stitching image is created, following two functions are valid
			int* GetStitchGridColumns() {return _piStitchGridCols;};
			int* GetStitchGridRows() {return _piStitchGridRows;};

			list<SubDevicCams>* GetSubDeviceInfo();

			///
			///	Clears all images from this layer
			///
			void ClearAllImages();

			void CameraCentersInY(double* pdCenY);
			void TriggerCentersInX(double* pdCenX);

			///
			///	Get the stitched buffer for the image... this needs to be filled in by alignment...
			///
			Image *GetStitchedImage(bool bRecreate = false);

			bool GetImagePatch(
				unsigned char* pBuf,
				unsigned int iPixelSpan,
				unsigned int iStartRowInCad,
				unsigned int iStartColInCad,
				unsigned int iRows,
				unsigned int iCols,
				FOVPreferSelected* pPreferSelectedFov);

			bool GetImagePatch(
				Image* pImage, 
				unsigned int iLeft,
				unsigned int iRight,
				unsigned int iTop,
				unsigned int iBottom,
				FOVPreferSelected* pPreferSelectedFov);

			void SetStitchedBuffer(unsigned char *pBuffer);

			void SetComponentHeightInfo(				
				unsigned char* pHeightBuf,		// Component height image buf
				unsigned int iHeightSpan,		// Component height image span
				double dHeightResolution,		// Height resolution in grey level (meter/grey level)
				double dPupilDistance);			// SIM pupil distance (meter))
			
			// For debug
			Image* GetGreyStitchedImage(bool bRecreate = false);
			void SetXShift(bool bValue);

		protected:
			/// Called from MosaicSet when a layer is added.
			void Initialize(MosaicSet *pMosaicSet, 
        		unsigned int numCameras,
				unsigned int numTriggers,
				bool bAlignWithCAD,
				bool bAlignWithFiducial,
				bool bFiducialBrighterThanBackground,
				bool bFiducialAllowNegativeMatch,
				unsigned int layerIndex,
				unsigned int deviceIndex);

			void CreateStitchedImageIfNecessary();
			void AllocateStitchedImageIfNecessary();
			bool CalculateStitchGrids();

			bool CalculateAllGridBoundary();
			bool CalculateGridBoundary(
				int iBeginInvTrig, int iEndInvTrig,
				int iBeginCam, int iEndCam,
				double* pdGridXBoundary, double* pdGridYBoundary);

			void CalculateFOVsForImagePatch(
				Image* pImage,	
				int iLeft, int iRight,
				int iTop, int iBottom,
				DRect worldRoi, bool bAjustForHeight,
				int* piStartInvTrig, int* piEndInvTrig,
				int* piStartCam, int* piEndCam,
				double* pdPatchGridXBoundary, double* pdPatchGridYBoundary);
			
			void CalPatchFOVRowBoundary(
				Image* pImage, 
				DRect worldRoi,
				unsigned int iTop,
				unsigned int iBottom,
				const double* pdPatchGridXBoundary,
				int iStartInvTrig,
				int iEndInvTrig,
				FOVPreferSelected* pPreferSelectedFov,
				int* piPixelRowBoundary);

			void CalPatchFOVColumnBoundary(
				Image* pImage, 
				DRect worldRoi,
				unsigned int iLeft,
				unsigned int iRight,
				const double* pdPatchGridYBoundary,
				int iStartCam,
				int iEndCam,
				FOVPreferSelected* pPreferSelectedFov,
				int* piPixelColBoundary);

			///
			///	Adds an image...
			///
			bool AddRawImage(unsigned char *pBuffer, unsigned int cameraIndex, unsigned int triggerIndex);
			bool AddYCrCbImage(unsigned char *pBuffer, unsigned int cameraIndex, unsigned int triggerIndex);

		private:
			bool _stitchedImageValid;
			unsigned int _numTriggers;
			unsigned int _numCameras;
			MosaicSet *_pMosaicSet;
			MosaicTile *_pTileArray;
			bool _bAlignWithCAD;
			bool _bAlignWithFiducial;
			bool _bFiducialBrighterThanBackground;
			bool _bFiducialAllowNegativeMatch;
			unsigned int _layerIndex;       // These indexes are used in alignment..
			unsigned int _deviceIndex;

			int* _piStitchGridRows;
			int* _piStitchGridCols; 
	
			Image *_pStitchedImage;

			// Image patch morph
			bool _bGridBoundaryValid;
				// Grid boundary x and y in world
			double* _pdGridXBoundary;	// organized with inverse trigger index (et. _pdGridXBoundary[0] and[1] for the last trigger)
			double* _pdGridYBoundary;

			// store the component heightinformation
			ComponentHeightInfo* _pHeightInfo;	

		// for Debug
			Image *_pGreyStitchedImage;
			bool _bXShift;
	};
}
