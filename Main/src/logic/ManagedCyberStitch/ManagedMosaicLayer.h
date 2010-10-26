#pragma once

#include "ManagedMosaicTile.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"

namespace MCyberStitch 
{
	public ref class ManagedMosaicLayer
	{
		public:
			ManagedMosaicLayer(CyberStitch::MosaicLayer *pMosaicLayer)
			{
				_pMosaicLayer = pMosaicLayer;
			}

			ManagedMosaicTile^ GetTile(int row, int column)
			{
				CyberStitch::MosaicTile *pTile = _pMosaicLayer->GetTile(row, column);
				return pTile == NULL?nullptr:gcnew ManagedMosaicTile(pTile);
			}

		private:
			CyberStitch::MosaicLayer *_pMosaicLayer;
	};
}
