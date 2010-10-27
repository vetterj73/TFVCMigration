#pragma once
#include "MosaicTile.h"

namespace MMosaicDM 
{
	///
	///	Simple Wrapper around unmanaged MosaicTile.  Only exposes what is necessary.
	///
	public ref class ManagedMosaicTile
	{
		public:
			///
			///	Constructor
			///
			ManagedMosaicTile(MosaicDM::MosaicTile * pMosaicTile)
			{
				_pMosaicTile = pMosaicTile;
			}
		
			///
			///	GetImageBuffer - returns the image buffer.
			///
			System::IntPtr GetImageBuffer()
			{
				return (System::IntPtr)_pMosaicTile->GetImageBuffer();
			}

			///
			///	SetImageBuffer - sets the image buffer.
			///
			void SetImageBuffer(System::IntPtr pImageBuffer)
			{
				_pMosaicTile->SetImageBuffer((unsigned char*)(void*)pImageBuffer);
			}

		private:
			MosaicDM::MosaicTile *_pMosaicTile;
	};
}

