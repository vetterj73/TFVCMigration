#pragma once

#include "LoggableObject.h"
using namespace LOGGER;

#include <vector>
#include <map>
using std::map;
using std::pair;
using std::vector;

namespace MosaicDM 
{
	class MosaicLayer;
	typedef vector<MosaicLayer*> LayerList;
	typedef LayerList::iterator LayerListIterator;

	class CorrelationFlags;
	typedef map< pair< int, int>, CorrelationFlags* > CorrelationFlagsMap;

	typedef void (*IMAGEADDED_CALLBACK)(int layerIndex, int cameraIndex, int triggerIndex, void* context);

	///
	///	MosaicSet is the top level object for Mosaic Data Model.  
	/// MosaicSet has 1 to N MosaicLayers.
	///
	class MosaicSet : public LoggableObject
	{
		public:

			///
			///	Constructor
			///
			/// \param objectWidthInMeters - how wide is the object we are imaging
			/// \param objectLengthInMeters what length is the object we are imaging
			/// \param imageWidthInPixels width of each image (tile) in pixels
			/// \param imageHeightInPixels height of each image (tile) in pixels
			/// \param imageStrideInPixels stride of each image (tile) in pixels
			/// \param nominalPixelSizeXInMeters 
			/// \param nominalPixelSizeYInMeters 
			MosaicSet(double objectWidthInMeters,
					  double objectLengthInMeters,
					  unsigned int imageWidthInPixels,
					  unsigned int imageHeightInPixels,
					  unsigned int imageStrideInPixels,
					  double nominalPixelSizeXInMeters,
					  double nominalPixelSizeYInMeters,
					  bool ownBuffers,
					  bool bBayerPattern,
					  int iBayerType);

			///
			///	Destructor
			///
			~MosaicSet();
		
			void RegisterImageAddedCallback(IMAGEADDED_CALLBACK pCallback, void* pContext);
			void UnregisterImageAddedCallback();

			///
			/// Adds a layer to a mosaic set
			///
			/// \param numCameras - number of cameras used for this layer
			/// \param numTriggers - number of triggers used for this layer
			/// \param bAlignWithCAD - should this layer be correlated against CAD?
			MosaicLayer *AddLayer(
				unsigned int numCameras,
				unsigned int numTriggers,
				bool bAlignWithCAD,
				bool bAlignWithFiducial,
				bool bFiducialBrighterThanBackground,
				bool bFiducialAllowNegativeMatch,
				unsigned int iDeviceIndex);
			
			///
			/// Gets a layer from the MosaicSet
			/// Returns null if the index is out of range.
			///
			/// \param index - Zero Based Index... 
			MosaicLayer *GetLayer(unsigned int index);

			///
			///	Getters for all basic Attributes
			///
			unsigned int GetNumMosaicLayers(){return (unsigned int)_layerList.size();}		
			unsigned int GetImageWidthInPixels(){return _imageWidth;}
			unsigned int GetImageHeightInPixels(){return _imageHeight;}
			unsigned int GetImageStrideInPixels(){return _imageStride;}
			unsigned int GetImageStrideInBytes(){return _imageStride;}
			double GetNominalPixelSizeX(){return _pixelSizeX;};
			double GetNominalPixelSizeY(){return _pixelSizeY;};
			double GetObjectWidthInMeters(){return _objectWidthInMeters;};
			double GetObjectLengthInMeters(){return _objectLengthInMeters;};
			unsigned int GetObjectWidthInPixels();
			unsigned int GetObjectLengthInPixels();
			bool HasOwnBuffers(){return _ownBuffers;}
			void SetOwnBuffers(bool bValue) { _ownBuffers = bValue;};
			bool IsBayerPattern(){return _bBayerPattern;}; 
			int GetBayerType(){return _iBayerType;};

			///
			///	Get the correlation flags associated with the certain layers
			///
			CorrelationFlags* GetCorrelationFlags(unsigned int layerX, unsigned int layerY);

			///
			///	Are all of the images from all layers collected?
			///
			bool HasAllImages();

			///
			///	Adds an image to the mosaic...
			///
			// Input buffer need to be Bayer or grayscale
			bool AddRawImage(unsigned char *pBuffer, unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex);
			// Input buffer need to be YCrCb
			bool AddYCrCbImage(unsigned char *pBuffer, unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex);

			///
			///	Clears all images from all layers in the mosaic.
			///
			void ClearAllImages();

			/// Saves all stitched images to a folder...
			bool SaveAllStitchedImagesToDirectory(string directoryName);

			/// Loads all stitched images from a folder...
			bool LoadAllStitchedImagesFromDirectory(string directoryName);

			/// Copies transforms from an existing mosaic set.
			bool CopyTransforms(MosaicSet *pMosaicSet);

			/// Copies transforms from an existing mosaic set.
			bool CopyBuffers(MosaicSet *pMosaicSet);

		private:
			unsigned int _imageWidth;
			unsigned int _imageHeight;
			unsigned int _imageStride;
			double _pixelSizeX;
			double _pixelSizeY;
			LayerList _layerList;
			double _objectWidthInMeters;
			double _objectLengthInMeters;
			IMAGEADDED_CALLBACK _registeredImageAddedCallback;
			void * _pCallbackContext;
			void FireImageAdded(unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex);
			CorrelationFlagsMap _correlationFlagsMap;
			bool _ownBuffers;
			bool _bBayerPattern;
			int _iBayerType;
	};
}
