// This is the main DLL file.

#include "stdafx.h"
#include "ManagedMosaic.h"

namespace MCyberStitch 
{

	ManagedMosaic::ManagedMosaic(int rows, int columns, int layers)
	{
		_pMosaic = new CyberStitch::Mosaic(rows, columns, layers);
	}

	// Finalizer...
	ManagedMosaic::!ManagedMosaic()
	{
		delete _pMosaic;
	}
}