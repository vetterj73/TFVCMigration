// CyberStitch.h

#pragma once
#include <vector>
using std::vector;

namespace MosaicDM 
{
	class MosaicLayer;
	typedef vector<MosaicLayer*> LayerList;
	typedef LayerList::iterator LayerListIterator;

	///
	///	MosaicSet is the top level object for Mosaic Data Model.  
	/// MosaicSet has 1 to N MosaicLayers.
	///
	class MosaicSet
	{
		public:

			///
			///	Constructor.  Need to call Initialize to setup the object
			///
			MosaicSet();

			///
			///	Destructor
			///
			~MosaicSet();

			///
			/// This is the main function to setup the entire mosaic data model.
			///
			/// \param rows # of rows in the mosaic
			/// \param columns # of columns in the mosaic
			/// \param imageWidthInPixels width of each image (tile) in pixels
			/// \param imageHeightInPixels height of each image (tile) in pixels
			/// \param imageStrideInPixels stride of each image (tile) in pixels
			/// \param bytesPerPixel # of bytes for each pixel
			/// \param overlapInMeters The overlap of each image. specified in meters.
			void Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int imageStrideInPixels, int bytesPerPixel, int overlapInMeters);
			
			///
			/// Adds a layer to a mosaic set
			///
			/// \param offsetInMeters - the initial offset (i.e. - where the first 
			MosaicLayer *AddLayer(double offsetInMM);
			
			///
			/// Gets a layer from the MosaicSet
			/// Returns null if the index is out of range.
			///
			/// \param index - Zero Based Index... 
			MosaicLayer *GetLayer(int index);

			///
			///	Resets everything to pre-initialized form.
			///
			void Reset();

			///
			///	Getters for all basic Attributes
			///
			int GetNumMosaicRows(){return _rows;}
			int GetNumMosaicColumns(){return _columns;}
			int GetNumMosaicLayers(){return _layerList.size();}		
			int GetImageWidthInPixels(){return _imageWidth;}
			int GetImageHeightInPixels(){return _imageHeight;}
			int GetImageStrideInPixels(){return _imageStride;}
			int GetImageStrideInBytes(){return _imageStride*_bytesPerPixel;}
			int GetBytesPerPixel(){return _bytesPerPixel;}
			int GetOverlapInMM(){return _overlapInMM;}
			int NumberOfTilesPerLayer();

			///
			///	Are all of the images from all layers collected?
			///
			bool HasAllImages();

		private:
			int _rows;
			int _columns;
			int _imageWidth;
			int _imageHeight;
			int _imageStride;
			int _bytesPerPixel;
			int _overlapInMM;
			LayerList _layerList;
	};
}
