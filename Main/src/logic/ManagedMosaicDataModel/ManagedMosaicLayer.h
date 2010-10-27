#pragma once

#include "ManagedMosaicTile.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"

namespace MMosaicDM 
{
	///
	///	Simple Wrapper around unmanaged MosaicLayer.  Only exposes what is necessary.
	///
	public ref class ManagedMosaicLayer
	{
		public:
			///
			///	Constructor
			///
			ManagedMosaicLayer(MosaicDM::MosaicLayer *pMosaicLayer)
			{
				_pMosaicLayer = pMosaicLayer;
			}

			///
			///	Gets a tile from the layer.
			///
			ManagedMosaicTile^ GetTile(int row, int column)
			{
				MosaicDM::MosaicTile *pTile = _pMosaicLayer->GetTile(row, column);
				return pTile == NULL?nullptr:gcnew ManagedMosaicTile(pTile);
			}

		private:
			MosaicDM::MosaicLayer *_pMosaicLayer;
	};
}
