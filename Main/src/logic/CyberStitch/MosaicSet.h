// CyberStitch.h

#pragma once
#include <vector>
using std::vector;


namespace CyberStitch 
{
	class MosaicLayer;
	
	typedef vector<MosaicLayer*> LayerList;
	typedef LayerList::iterator LayerListIterator;

	//
	//	MosaicSet is the top level object for correlation and stitching.
	//
	class MosaicSet
	{
		public:
			MosaicSet();
			~MosaicSet();

			void Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int imageStrideInPixels, int bytesPerPixel, int overlapInMM);
			MosaicLayer *AddLayer(double offsetInMM);
			MosaicLayer *GetLayer(int index);

			void Reset();

			int GetNumMosaicRows(){return _rows;}
			int GetNumMosaicColumns(){return _columns;}
			int GetNumMosaicLayers(){return _layerList.size();}
			
			int GetImageWidthInPixels(){return _imageWidth;}
			int GetImageHeightInPixels(){return _imageHeight;}
			int GetImageStrideInPixels(){return _imageStride;}
			int GetImageStrideInBytes(){return _imageStride*_bytesPerPixel;}
			int GetBytesPerPixel(){return _bytesPerPixel;}
			int GetOverlapInMM(){return _overlapInMM;}

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
