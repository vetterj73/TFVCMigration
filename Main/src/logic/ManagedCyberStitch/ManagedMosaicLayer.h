#pragma once

#include "MosaicLayer.h"

namespace MCyberStitch 
{
	public ref class ManagedMosaicLayer
	{
		public:
			ManagedMosaicLayer(CyberStitch::MosaicLayer *pMosaicLayer)
			{
				_pMosaicLayer = pMosaicLayer;
			}

		private:
			CyberStitch::MosaicLayer *_pMosaicLayer;
	};
}
