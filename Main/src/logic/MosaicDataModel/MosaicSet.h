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
					  int imageWidthInPixels,
					  int imageHeightInPixels,
					  int imageStrideInPixels,
					  double nominalPixelSizeXInMeters,
					  double nominalPixelSizeYInMeters);

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
				int numCameras,
				int numTriggers,
				bool bAlignWithCAD,
				bool bAlignWithFiducial);
			
			///
			/// Gets a layer from the MosaicSet
			/// Returns null if the index is out of range.
			///
			/// \param index - Zero Based Index... 
			MosaicLayer *GetLayer(int index);

			///
			///	Getters for all basic Attributes
			///
			int GetNumMosaicLayers(){return (int)_layerList.size();}		
			int GetImageWidthInPixels(){return _imageWidth;}
			int GetImageHeightInPixels(){return _imageHeight;}
			int GetImageStrideInPixels(){return _imageStride;}
			int GetImageStrideInBytes(){return _imageStride;}
			double GetNominalPixelSizeX(){return _pixelSizeX;};
			double GetNominalPixelSizeY(){return _pixelSizeY;};
			double GetObjectWidthInMeters(){return _objectWidthInMeters;};
			double GetObjectLengthInMeters(){return _objectLengthInMeters;};

			///
			///	Get the correlation flags associated with the certain layers
			///
			CorrelationFlags* GetCorrelationFlags(int layerX, int layerY);

			///
			///	Are all of the images from all layers collected?
			///
			bool HasAllImages();

			///
			///	Adds an image to the mosaic...
			///
			bool AddImage(unsigned char *pBuffer, int layerIndex, int cameraIndex, int triggerIndex);

			///
			///	Clears all images from all layers in the mosaic.
			///
			void ClearAllImages();

		private:
			int _imageWidth;
			int _imageHeight;
			int _imageStride;
			double _pixelSizeX;
			double _pixelSizeY;
			LayerList _layerList;
			double _objectWidthInMeters;
			double _objectLengthInMeters;
			IMAGEADDED_CALLBACK _registeredImageAddedCallback;
			void * _pCallbackContext;
			void FireImageAdded(int layerIndex, int cameraIndex, int triggerIndex);
			CorrelationFlagsMap _correlationFlagsMap;
	};
}
