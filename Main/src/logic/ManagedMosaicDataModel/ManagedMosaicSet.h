// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"
#include "ManagedMosaicLayer.h"
#include "ManagedCorrelationFlags.h"
#include "LoggableManagedObject.h"

using namespace System;
using namespace MLOGGER;
namespace MMosaicDM 
{
	public delegate void ImageAddedDelegate(int, int, int);
	
	///
	///	Simple Wrapper around unmanaged MosaicSet.  Only exposes what is necessary.
	/// NOTE:  This only works with 8 bit images!
	///
	public ref class ManagedMosaicSet : public MLoggableObject
	{
		public:

			///
			///	Constructor - See MosaicSet constructor for details.
			///
			ManagedMosaicSet(double objectWidthInMeters,
					  double objectLengthInMeters,
					  int imageWidthInPixels,
					  int imageHeightInPixels,
					  int imageStrideInPixels,
					  double pixelSizeXInMeters,
					  double pixelSizeYInMeters)
			{
				_pMosaicSet = new MosaicDM::MosaicSet(
						objectWidthInMeters,
						objectLengthInMeters,
						imageWidthInPixels,
						imageHeightInPixels,
						imageStrideInPixels,
						pixelSizeXInMeters,
						pixelSizeYInMeters
					);

				// This sets up the Logging interface from unmanaged...
				SetLoggableObject(_pMosaicSet);

				_imageAddedDelegate = gcnew ImageAddedDelegate(this, &MMosaicDM::ManagedMosaicSet::RaiseImageAdded); 
				_pMosaicSet->RegisterImageAddedCallback((MosaicDM::IMAGEADDED_CALLBACK)Marshal::GetFunctionPointerForDelegate(_imageAddedDelegate).ToPointer(), NULL);	
			}


			///
			///	Finalizer (deletes the unmanaged pointer).
			///
			!ManagedMosaicSet()
			{
				delete _pMosaicSet;
			}

			///
			///	Destructor - not called, but defining it avoids warnngs (and I verified that
			/// the finalizer is called.
			///
			~ManagedMosaicSet()
			{
			//	delete _pMosaicSet;
			}		
	
			///
			///	Adds a layer (see unmanaged MosaicSet for details)
			///
			ManagedMosaicLayer ^AddLayer(double cameraOffsetInMeters, 
									double triggerOffsetInMeters,
        							int numCameras,
									double cameraOverlapInMeters,
									int numTriggers,
									double triggerOverlapInMeters,
									bool correlateWithCAD)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->AddLayer(
					cameraOffsetInMeters, 
					triggerOffsetInMeters,
        			numCameras,
					cameraOverlapInMeters,
					numTriggers,
					triggerOverlapInMeters,
					correlateWithCAD);
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

			/// \internal
			property System::IntPtr UnmanagedMosaicSet
			{
				System::IntPtr get() { return safe_cast<System::IntPtr>(_pMosaicSet); }
			}

		private:
			MosaicDM::MosaicSet *_pMosaicSet;
		
			ImageAddedDelegate ^_imageAddedDelegate;
			void RaiseImageAdded(int layerIndex, int cameraIndex, int triggerIndex)
			{
				OnImageAdded(layerIndex, cameraIndex, triggerIndex);
			}
	};
}
