// This is the main DLL file.

#include "stdafx.h"
#include "ManagedMosaicSet.h"

namespace MCyberStitch 
{
	ManagedMosaicSet::ManagedMosaicSet()
	{
		_pMosaicSet = new CyberStitch::MosaicSet();
	}

	// Finalizer...
	ManagedMosaicSet::!ManagedMosaicSet()
	{
		delete _pMosaicSet;
	}

	void ManagedMosaicSet::Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int overlapInMM)
	{
		_pMosaicSet->Initialize(rows, columns, imageWidthInPixels, imageHeightInPixels, overlapInMM);
	}
}