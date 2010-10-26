// This is the main DLL file.

#include "stdafx.h"
#include "MosaicSet.h"
#include "MosaicLayer.h"

namespace CyberStitch 
{
	MosaicSet::MosaicSet()
	{
		_rows = 0;
		_columns = 0;
	}

	MosaicSet::~MosaicSet()
	{
		Reset();
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

	void MosaicSet::Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int overlapInMM)
	{
		Reset();

		_rows = rows;
		_columns = columns;
		_imageWidth = imageWidthInPixels;
		_imageHeight = imageHeightInPixels;
		_overlapInMM = overlapInMM;
	}

	void MosaicSet::Reset()
	{
		for(int i=0; i<_layerList.size(); i++)
			delete _layerList[i];

		_layerList.clear();
		_rows = 0;
		_columns = 0;
		_imageWidth = 0;
		_imageHeight = 0;
		_overlapInMM = 0;
	}
}