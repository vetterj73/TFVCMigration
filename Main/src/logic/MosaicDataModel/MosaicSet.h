// CyberStitch.h

#pragma once

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
	class MosaicSet
	{
		public:

			///
			///	Constructor
			///
			/// \param objectWidthInMeters - how wide is the object we are imaging
			/// \param objectLengthInMeters what length is the object we are imaging
			/// \param numCameras (columns) in mosaic
			/// \param cameraOverlapInMeters overlap between cameras
			/// \param numTriggers # of triggers in the mosaic
			/// \param triggerOverlapInMeters overlap between triggers
			/// \param imageWidthInPixels width of each image (tile) in pixels
			/// \param imageHeightInPixels height of each image (tile) in pixels
			/// \param imageStrideInPixels stride of each image (tile) in pixels
			/// \param overlapInMeters The overlap of each image. specified in meters.
			/// \param pixelSizeXInMeters - size of pixel in X direction.
			/// \param pixelSizeYInMeters - size of pixel in Y direction.			
			MosaicSet(double objectWidthInMeters,
					  double objectLengthInMeters,
					  int numCameras,
					  double cameraOverlapInMeters,
					  int numTriggers,
					  double triggerOverlapInMeters,
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
			/// \param offsetInMeters - the initial offset from edge of image 
			MosaicLayer *AddLayer(double offsetInMeters);
			
			///
			/// Gets a layer from the MosaicSet
			/// Returns null if the index is out of range.
			///
			/// \param index - Zero Based Index... 
			MosaicLayer *GetLayer(int index);

			///
			///	Getters for all basic Attributes
			///
			int GetNumTriggers(){return _triggers;}
			int GetNumCameras(){return _cameras;}
			int GetNumMosaicLayers(){return _layerList.size();}		
			int GetImageWidthInPixels(){return _imageWidth;}
			int GetImageHeightInPixels(){return _imageHeight;}
			int GetImageStrideInPixels(){return _imageStride;}
			int GetImageStrideInBytes(){return _imageStride;}
			int GetCameraOverlapInMeters(){return _cameraOverlap;}
			int GetTriggerOverlapInMeters(){return _triggerOverlap;}
			int NumberOfTilesPerLayer();
			double GetNominalPixelSizeX(){return _pixelSizeX;};
			double GetNominalPixelSizeY(){return _pixelSizeY;};
			double GetObjectWidthInMeters(){return _objectWidthInMeters;};
			double GetObjectLengthInMeters(){return _objectLengthInMeters;};

			///
			///	Get the correlation flags associated with the current layers
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



		private:
			int _triggers;
			int _cameras;
			int _imageWidth;
			int _imageHeight;
			int _imageStride;
			double _triggerOverlap;
			double _cameraOverlap;
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
