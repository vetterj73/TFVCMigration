// ManagedCyberStitch.h

#pragma once

#include "Mosaic.h"

using namespace System;
namespace MCyberStitch 
{
	public ref class ManagedMosaic
	{
		public:
			ManagedMosaic(int rows, int columns, int layers);
			!ManagedMosaic();

		private:
			ManagedMosaic(){};
			CyberStitch::Mosaic *_pMosaic;
	};
}
