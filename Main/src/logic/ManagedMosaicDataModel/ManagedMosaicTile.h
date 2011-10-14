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
				return (System::IntPtr)_pMosaicTile->GetImagPtr()->GetBuffer();
			}

			bool Bayer2Lum(int iType)
			{
				return _pMosaicTile->Bayer2Lum(iType);
			}

			///
			///	Sets the parameters needed for transform.  If this function isn't called,
			/// nominal values will be used.
			///
			void SetTransformParameters(double pixelSizeXInMeters, double pixelSizeYInMeters, 
				double rotation,
				double centerOffsetXInMeters, double centerOffsetYInMeters)
			{
				_pMosaicTile->SetTransformParameters(pixelSizeXInMeters, pixelSizeYInMeters,
					rotation,
					centerOffsetXInMeters, centerOffsetYInMeters);
			}

		private:
			MosaicDM::MosaicTile *_pMosaicTile;
	};
}

