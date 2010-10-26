#pragma once
#include "MosaicTile.h"

namespace MCyberStitch 
{
	public ref class ManagedMosaicTile
	{
		public:
			ManagedMosaicTile(CyberStitch::MosaicTile * pMosaicTile)
			{
				_pMosaicTile = pMosaicTile;
			}
		
		private:
			CyberStitch::MosaicTile *_pMosaicTile;
	};
}

