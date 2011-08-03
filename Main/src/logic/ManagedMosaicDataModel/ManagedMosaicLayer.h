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
			ManagedMosaicTile^ GetTile(unsigned int cameraIndex, unsigned int triggerIndex)
			{
				MosaicDM::MosaicTile *pTile = _pMosaicLayer->GetTile(cameraIndex, triggerIndex);
				return pTile == NULL?nullptr:gcnew ManagedMosaicTile(pTile);
			}

			void ClearAllImages()
			{
				return _pMosaicLayer->ClearAllImages();
			}

			int GetNumberOfTriggers(){return _pMosaicLayer->GetNumberOfTriggers();};
			int GetNumberOfCameras(){return  _pMosaicLayer->GetNumberOfCameras();};

			bool IsAlignWithCad() {return _pMosaicLayer->IsAlignWithCad();};
			void SetAlignWithCad(bool bAlignWithCad) { _pMosaicLayer->SetAlignWithCad(bAlignWithCad);}; 
			
			bool IsAlignWithFiducial() {return _pMosaicLayer->IsAlignWithFiducial();};
			void SetAlignWithFiducial(bool bAlignWithFiducial) { _pMosaicLayer->SetAlignWithFiducial(bAlignWithFiducial);}; 


			System::IntPtr GetStitchedBuffer()
			{
				return GetStitchedBuffer(false);
			}

			System::IntPtr GetStitchedBuffer(bool bRecreate)
			{
				Image* stitchedImage = _pMosaicLayer->GetStitchedImage(bRecreate);
				return (System::IntPtr)stitchedImage->GetBuffer();
			}

			System::IntPtr GetStitchedBufferWithHeight(System::IntPtr pHeighBuf, double dHeightResolution, double dPupilDistance)
			{
				return GetStitchedBufferWithHeight(pHeighBuf, dHeightResolution, dPupilDistance, false);
			}

			System::IntPtr GetStitchedBufferWithHeight(System::IntPtr pHeighBuf, double dHeightResolution, double dPupilDistance, bool bRecreate)
			{
				Image* stitchedImage = _pMosaicLayer->GetStitchedImageWithHeight(
					(unsigned char*)(void*)pHeighBuf, dHeightResolution, dPupilDistance,
					bRecreate);
				return (System::IntPtr)stitchedImage->GetBuffer();
			}

		private:
			MosaicDM::MosaicLayer *_pMosaicLayer;
	};
}
