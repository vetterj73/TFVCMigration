// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"

using namespace System;
namespace MCyberStitch 
{
	public ref class ManagedMosaicSet
	{
		public:
			ManagedMosaicSet();
			!ManagedMosaicSet();
			void Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int overlapInMM);

		private:
			CyberStitch::MosaicSet *_pMosaicSet;
	};
}
