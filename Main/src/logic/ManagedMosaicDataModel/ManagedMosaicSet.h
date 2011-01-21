// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"
#include "ManagedMosaicLayer.h"
#include "ManagedCorrelationFlags.h"
#include "ManagedLoggableObject.h"

using namespace System;
using namespace MLOGGER;

/// \mainpage
///
/// \image html SIM.png CyberOptics' SIM Sensor
///
/// \section intro Introduction
/// This document describes the API interface for CyberStitch.
/// The CyberStitch Components are used to take a set of images (normally from a SIM Device)
/// and stitch them together.  The images can be from one or more Devices.  CAD and Mask
/// information can also be used to help the stitching process.
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
			ManagedMosaicLayer ^AddLayer(
        		int numCameras,
				int numTriggers,
				bool bAlignWithCAD,
				bool bAlignWithFiducial)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->AddLayer(
        			numCameras,
					numTriggers,
					bAlignWithCAD,
					bAlignWithFiducial);
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

			bool HasAllImages()
			{
				return _pMosaicSet->HasAllImages();
			}
			
			bool AddImage(System::IntPtr pBuffer, int layerIndex, int cameraIndex, int triggerIndex)
			{
				return _pMosaicSet->AddImage((unsigned char*)(void*)pBuffer, layerIndex, cameraIndex, triggerIndex);
			}

			void ClearAllImages()
			{
				_pMosaicSet->ClearAllImages();
			}

			event ImageAddedDelegate^ OnImageAdded;

			/// \internal
			property System::IntPtr UnmanagedMosaicSet
			{
				System::IntPtr get() { return safe_cast<System::IntPtr>(_pMosaicSet); }
			}

			int GetNumMosaicLayers(){ return _pMosaicSet->GetNumMosaicLayers();}

			double GetObjectWidthInMeters(){return _pMosaicSet->GetObjectWidthInMeters();}
			double GetObjectLengthInMeters(){return _pMosaicSet->GetObjectLengthInMeters();}

		private:
			MosaicDM::MosaicSet *_pMosaicSet;
		
			ImageAddedDelegate ^_imageAddedDelegate;
			void RaiseImageAdded(int layerIndex, int cameraIndex, int triggerIndex)
			{
				OnImageAdded(layerIndex, cameraIndex, triggerIndex);
			}
	};
}
