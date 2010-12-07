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
			ManagedMosaicTile^ GetTile(int cameraIndex, int triggerIndex)
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

			bool IsUseCad() {return _pMosaicLayer->IsUseCad();};
			void SetUseCad(bool bUseCad) { _pMosaicLayer->SetUseCad(bUseCad);}; 

		private:
			MosaicDM::MosaicLayer *_pMosaicLayer;
	};
}
