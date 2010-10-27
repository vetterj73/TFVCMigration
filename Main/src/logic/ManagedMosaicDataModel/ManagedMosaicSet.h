// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"
#include "ManagedMosaicLayer.h"

using namespace System;
namespace MMosaicDM 
{
	public ref class ManagedMosaicSet
	{
		public:
			ManagedMosaicSet()
			{
				_pMosaicSet = new MosaicDM::MosaicSet();
			}

			!ManagedMosaicSet()
			{
				delete _pMosaicSet;
			}
			
			void Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int imageStrideInPixels, int bytesPerPixel, int overlapInMM)
			{
				_pMosaicSet->Initialize(rows, columns, imageWidthInPixels, imageHeightInPixels, imageStrideInPixels, bytesPerPixel, overlapInMM);	
			}

			void Reset()
			{
				_pMosaicSet->Reset();
			}

			ManagedMosaicLayer ^AddLayer(double offsetInMM)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->AddLayer(offsetInMM);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

			ManagedMosaicLayer ^GetLayer(int index)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->GetLayer(index);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

		private:
			MosaicDM::MosaicSet *_pMosaicSet;
	};
}
