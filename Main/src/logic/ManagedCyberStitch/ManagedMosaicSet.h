// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"
#include "ManagedMosaicLayer.h"

using namespace System;
namespace MCyberStitch 
{
	public ref class ManagedMosaicSet
	{
		public:
			ManagedMosaicSet()
			{
				_pMosaicSet = new CyberStitch::MosaicSet();
			}

			!ManagedMosaicSet()
			{
				delete _pMosaicSet;
			}
			
			void Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int overlapInMM)
			{
				_pMosaicSet->Initialize(rows, columns, imageWidthInPixels, imageHeightInPixels, overlapInMM);	
			}

			void Reset()
			{
				_pMosaicSet->Reset();
			}

			ManagedMosaicLayer ^AddLayer(double offsetInMM)
			{
				CyberStitch::MosaicLayer* pLayer = _pMosaicSet->AddLayer(offsetInMM);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

			ManagedMosaicLayer ^GetLayer(int index)
			{
				CyberStitch::MosaicLayer* pLayer = _pMosaicSet->GetLayer(index);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

		private:
			CyberStitch::MosaicSet *_pMosaicSet;
	};
}
