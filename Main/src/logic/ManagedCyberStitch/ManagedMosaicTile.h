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
		
			System::IntPtr GetImageBuffer()
			{
				return (System::IntPtr)_pMosaicTile->GetImageBuffer();
			}

			void SetImageBuffer(System::IntPtr pImageBuffer)
			{
				_pMosaicTile->SetImageBuffer((unsigned char*)(void*)pImageBuffer);
			}

		private:
			CyberStitch::MosaicTile *_pMosaicTile;
	};
}

