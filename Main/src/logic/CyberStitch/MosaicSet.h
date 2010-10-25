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

			void Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int overlapInMM);
			void AddLayer(double offsetInMM);

			void Reset();
			MosaicLayer *GetLayer(int index);

			int GetNumRows(){return _rows;}
			int GetNumColumns(){return _columns;}
			int GetNumLayers(){return _layerList.size();}
			int GetImageWidth(){return _imageWidth;}
			int GetImageHeight(){return _imageHeight;}
			int GetOverlapInMM(){return _overlapInMM;}

		private:
			int _rows;
			int _columns;
			int _imageWidth;
			int _imageHeight;
			int _overlapInMM;
			LayerList _layerList;
	};
}
