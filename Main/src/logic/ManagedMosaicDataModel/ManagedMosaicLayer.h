#pragma once

#include "ManagedMosaicTile.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"

namespace MMosaicDM 
{
	public ref class ManagedMosaicLayer
	{
		public:
			ManagedMosaicLayer(MosaicDM::MosaicLayer *pMosaicLayer)
			{
				_pMosaicLayer = pMosaicLayer;
			}

			ManagedMosaicTile^ GetTile(int row, int column)
			{
				MosaicDM::MosaicTile *pTile = _pMosaicLayer->GetTile(row, column);
				return pTile == NULL?nullptr:gcnew ManagedMosaicTile(pTile);
			}

		private:
			MosaicDM::MosaicLayer *_pMosaicLayer;
	};
}
