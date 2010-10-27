#pragma once
#include "MosaicTile.h"

namespace MMosaicDM 
{
	public ref class ManagedMosaicTile
	{
		public:
			ManagedMosaicTile(MosaicDM::MosaicTile * pMosaicTile)
			{
				_pMosaicTile = pMosaicTile;
			}
		
			System::IntPtr GetImageBuffer()
			{
				return (System::IntPtr)_pMosaicTile->GetImageBuffer();
			}

			void SetImageBuffer(System::IntPtr pImageBuffer)
			{
				_pMosaicTile->SetImageBuffer((unsigned char*)(void*)pImageBuffer);
			}

		private:
			MosaicDM::MosaicTile *_pMosaicTile;
	};
}

