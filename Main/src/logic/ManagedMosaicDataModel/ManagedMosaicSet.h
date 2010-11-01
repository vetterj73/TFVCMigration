// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"
#include "ManagedMosaicLayer.h"
#include "ManagedCorrelationFlags.h"

using namespace System;
using namespace System::Runtime::InteropServices;
namespace MMosaicDM 
{

	public delegate void ImageAddedDelegate(int, int, int);
	
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

				_ImageAddedDelegate = gcnew ImageAddedDelegate(this, &MMosaicDM::ManagedMosaicSet::RaiseImageAdded); 
				_pMosaicSet->RegisterImageAddedCallback((MosaicDM::IMAGEADDED_CALLBACK)Marshal::GetFunctionPointerForDelegate(_ImageAddedDelegate).ToPointer(), NULL);
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

			///
			///	Gets a CorrelationFlags structure to fill in.
			///
			ManagedCorrelationFlags ^GetCorrelationSet(int layerX, int layerY)
			{		
				MosaicDM::CorrelationFlags* pCF = _pMosaicSet->GetCorrelationFlags(layerX, layerY);
				return pCF == NULL?nullptr:gcnew ManagedCorrelationFlags(pCF);
			}

			bool AddImage(System::IntPtr pBuffer, int layerIndex, int cameraIndex, int triggerIndex)
			{
				return _pMosaicSet->AddImage((unsigned char*)(void*)pBuffer, layerIndex, cameraIndex, triggerIndex);
			}

			event ImageAddedDelegate^ OnImageAdded;

		private:
			MosaicDM::MosaicSet *_pMosaicSet;
			ImageAddedDelegate ^_ImageAddedDelegate;
		
			void RaiseImageAdded(int layerIndex, int cameraIndex, int triggerIndex)
			{
				OnImageAdded(layerIndex, cameraIndex, triggerIndex);
			}
	};
}
