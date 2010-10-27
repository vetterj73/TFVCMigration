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
			///	Constructor - See MosaicSet constructor for details.
			///
			ManagedMosaicSet(int numRowsInMosaic,
					  double rowOverlapInMeters,
					  int numColumnsInMosaic,
					  double columnOverlapInMeters,
					  int imageWidthInPixels,
					  int imageHeightInPixels,
					  int imageStrideInPixels,
					  int bytesPerPixel,
					  double pixelSizeXInMeters,
					  double pixelSizeYInMeters)
			{
				_pMosaicSet = new MosaicDM::MosaicSet(
						numRowsInMosaic,
						rowOverlapInMeters,
						numColumnsInMosaic,
						columnOverlapInMeters,
						imageWidthInPixels,
						imageHeightInPixels,
						imageStrideInPixels,
						bytesPerPixel,
						pixelSizeXInMeters,
						pixelSizeYInMeters
					);
			}

			///
			///	Finalizer (deletes the unmanaged pointer).
			///
			!ManagedMosaicSet()
			{
				delete _pMosaicSet;
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
