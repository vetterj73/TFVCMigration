// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"
#include "ManagedMosaicLayer.h"

using namespace System;
namespace MMosaicDM 
{
	///
	///	Simple Wrapper around unmanaged MosaicSet.  Only exposes what is necessary.
	///
	public ref class ManagedMosaicSet
	{
		public:

			///
			///	Constructor
			///
			ManagedMosaicSet()
			{
				_pMosaicSet = new MosaicDM::MosaicSet();
			}

			///
			///	Finalizer (deletes the unmanaged pointer).
			///
			!ManagedMosaicSet()
			{
				delete _pMosaicSet;
			}
			
			///
			///	Initialize - see unmanaged MosaicSet for details.
			///
			void Initialize(int rows, int columns, int imageWidthInPixels, int imageHeightInPixels, int imageStrideInPixels, int bytesPerPixel, int overlapInMeters)
			{
				_pMosaicSet->Initialize(rows, columns, imageWidthInPixels, imageHeightInPixels, imageStrideInPixels, bytesPerPixel, overlapInMeters);	
			}

			///
			///	Resets to pre-initialized state.
			///
			void Reset()
			{
				_pMosaicSet->Reset();
			}

			///
			///	Adds a layer (see unmanaged MosaicSet for details)
			///
			ManagedMosaicLayer ^AddLayer(double offsetInMM)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->AddLayer(offsetInMM);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

			///
			///	Gets a layer (see unmanaged MosaicSet for details)
			///
			ManagedMosaicLayer ^GetLayer(int index)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->GetLayer(index);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

		private:
			MosaicDM::MosaicSet *_pMosaicSet;
	};
}
