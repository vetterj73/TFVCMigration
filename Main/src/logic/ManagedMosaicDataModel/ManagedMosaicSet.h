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
			ManagedMosaicSet(int numCameras,
					  double cameraOverlapInMeters,
					  int numTriggers,
					  double triggerOverlapInMeters,
					  int imageWidthInPixels,
					  int imageHeightInPixels,
					  int imageStrideInPixels,
					  int bytesPerPixel,
					  double pixelSizeXInMeters,
					  double pixelSizeYInMeters)
			{
				_pMosaicSet = new MosaicDM::MosaicSet(
						numCameras,
						cameraOverlapInMeters,
						numTriggers,
						triggerOverlapInMeters,
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
			ManagedMosaicLayer ^AddLayer(double offsetInMeters)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->AddLayer(offsetInMeters);
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
