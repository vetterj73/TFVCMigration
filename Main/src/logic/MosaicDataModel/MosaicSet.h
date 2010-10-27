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
			///	Constructor
			///
			/// \param numRowsInMosaic # of rows in the mosaic
			/// \param rowOverlapInMeters overlap between rows
			/// \param numColumnsInMosaic # of columns in the mosaic
			/// \param columnOverlapInMeters overlap between columns
			/// \param imageWidthInPixels width of each image (tile) in pixels
			/// \param imageHeightInPixels height of each image (tile) in pixels
			/// \param imageStrideInPixels stride of each image (tile) in pixels
			/// \param bytesPerPixel # of bytes for each pixel
			/// \param overlapInMeters The overlap of each image. specified in meters.
			/// \param pixelSizeXInMeters - size of pixel in X direction.
			/// \param pixelSizeYInMeters - size of pixel in Y direction.			
			MosaicSet(int numRowsInMosaic,
					  double rowOverlapInMeters,
					  int numColumnsInMosaic,
					  double columnOverlapInMeters,
					  int imageWidthInPixels,
					  int imageHeightInPixels,
					  int imageStrideInPixels,
					  int bytesPerPixel,
					  double pixelSizeXInMeters,
					  double pixelSizeYInMeters);

			///
			///	Destructor
			///
			~MosaicSet();
		
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
			int GetNumMosaicRows(){return _rows;}
			int GetNumMosaicColumns(){return _columns;}
			int GetNumMosaicLayers(){return _layerList.size();}		
			int GetImageWidthInPixels(){return _imageWidth;}
			int GetImageHeightInPixels(){return _imageHeight;}
			int GetImageStrideInPixels(){return _imageStride;}
			int GetImageStrideInBytes(){return _imageStride*_bytesPerPixel;}
			int GetBytesPerPixel(){return _bytesPerPixel;}
			int GetColumnOverlapInMeters(){return _columnOverlap;}
			int GetRowOverlapInMeters(){return _rowOverlap;}
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
			double _rowOverlap;
			double _columnOverlap;
			double _pixelSizeX;
			double _pixelSizeY;
			LayerList _layerList;
	};
}
