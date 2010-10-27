// This is the main DLL file.

#include "stdafx.h"
#include "MosaicSet.h"
#include "MosaicLayer.h"

namespace MosaicDM 
{
	MosaicSet::MosaicSet(int numRowsInMosaic,
			  double rowOverlapInMeters,
			  int numColumnsInMosaic,
			  double columnOverlapInMeters,
			  int imageWidthInPixels,
			  int imageHeightInPixels,
			  int imageStrideInPixels,
			  int bytesPerPixel,
			  double pixelSizeXInMeters,
			  double pixelSizeYInMeters)
	{
		_rows = numRowsInMosaic;
		_columns = numColumnsInMosaic;
		_rowOverlap = rowOverlapInMeters;
		_columnOverlap = columnOverlapInMeters;
		_imageWidth = imageWidthInPixels;
		_imageHeight = imageHeightInPixels;
		_imageStride = imageStrideInPixels;
		_bytesPerPixel = bytesPerPixel;
		_pixelSizeX = pixelSizeXInMeters;
		_pixelSizeY = pixelSizeYInMeters;
	}

	MosaicSet::~MosaicSet()
	{
		for(int i=0; i<_layerList.size(); i++)
			delete _layerList[i];

		_layerList.clear();
	}

	MosaicLayer *MosaicSet::GetLayer(int index)
	{
		if(index<0 || index >= _layerList.size())
			return NULL;

		return _layerList[index];
	}

	MosaicLayer * MosaicSet::AddLayer(double offsetInMM)
	{
		MosaicLayer *pML = new MosaicLayer();
		pML->Initialize(this, offsetInMM);
		_layerList.push_back(pML);
		return pML;
	}

	int MosaicSet::NumberOfTilesPerLayer()
	{
		return GetNumMosaicRows()*GetNumMosaicColumns();
	}
	
	bool MosaicSet::HasAllImages()
	{
		int numTiles = NumberOfTilesPerLayer();
		for(int i=0; i<_layerList.size(); i++)
			if(!_layerList[i]->HasAllImages())
				return false;

		return true;
	}
}