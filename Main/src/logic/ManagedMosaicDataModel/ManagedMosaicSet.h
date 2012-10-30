// ManagedCyberStitch.h

#pragma once

#include "MosaicSet.h"
#include "ManagedMosaicLayer.h"
#include "ManagedCorrelationFlags.h"
using namespace System::Runtime::InteropServices;

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
	public delegate void ImageAddedDelegate(unsigned int, unsigned int, unsigned int);
	
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
					  unsigned int imageWidthInPixels,
					  unsigned int imageHeightInPixels,
					  unsigned int imageStrideInPixels,
					  double pixelSizeXInMeters,
					  double pixelSizeYInMeters,
					  bool ownBuffers,
					  bool bBayerPattern,
					  int iBayerType,
					  bool bSkipDemosaic)
			{
				_pMosaicSet = new MosaicDM::MosaicSet(
						objectWidthInMeters,
						objectLengthInMeters,
						imageWidthInPixels,
						imageHeightInPixels,
						imageStrideInPixels,
						pixelSizeXInMeters,
						pixelSizeYInMeters,
						ownBuffers,
						bBayerPattern,
						iBayerType,
						bSkipDemosaic);

				// This sets up the Logging interface from unmanaged...
				SetLoggableObject((System::IntPtr)_pMosaicSet);

				_imageAddedDelegate = gcnew ImageAddedDelegate(this, &MMosaicDM::ManagedMosaicSet::RaiseImageAdded); 
				_pMosaicSet->RegisterImageAddedCallback((MosaicDM::IMAGEADDED_CALLBACK)Marshal::GetFunctionPointerForDelegate(_imageAddedDelegate).ToPointer(), NULL);	
			}

			~ManagedMosaicSet()
			{
				delete _pMosaicSet;
				_pMosaicSet = NULL;
			}

			///
			///	Finalizer (deletes the unmanaged pointer).
			///
			!ManagedMosaicSet()
			{
				delete _pMosaicSet;
				_pMosaicSet = NULL;
			}

			///
			///	Adds a layer (see unmanaged MosaicSet for details)
			///
			ManagedMosaicLayer ^AddLayer(
        		unsigned int numCameras,
				unsigned int numTriggers,
				bool bAlignWithCAD,
				bool bAlignWithFiducial,
				bool bFiducialBrighterThanBackground,
				bool bFiducialAllowNegativeMatch,
				unsigned int iDeviceIndex)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->AddLayer(
        			numCameras,
					numTriggers,
					bAlignWithCAD,
					bAlignWithFiducial,
					bFiducialBrighterThanBackground,
					bFiducialAllowNegativeMatch,
					iDeviceIndex);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

			///
			///	Gets a layer (see unmanaged MosaicSet for details)
			///
			ManagedMosaicLayer ^GetLayer(unsigned int index)
			{
				MosaicDM::MosaicLayer* pLayer = _pMosaicSet->GetLayer(index);
				return pLayer == NULL?nullptr:gcnew ManagedMosaicLayer(pLayer);
			}

			///
			///	Gets a CorrelationFlags structure to fill in.
			///
			ManagedCorrelationFlags ^GetCorrelationSet(unsigned int layerX, unsigned int layerY)
			{		
				MosaicDM::CorrelationFlags* pCF = _pMosaicSet->GetCorrelationFlags(layerX, layerY);
				return pCF == NULL?nullptr:gcnew ManagedCorrelationFlags(pCF);
			}

			bool HasAllImages()
			{
				return _pMosaicSet->HasAllImages();
			}
			
			bool AddRawImage(System::IntPtr pBuffer, unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex)
			{
				return _pMosaicSet->AddRawImage((unsigned char*)(void*)pBuffer, layerIndex, cameraIndex, triggerIndex);
			}

			bool AddYCrCbImage(System::IntPtr pBuffer, unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex)
			{
				return _pMosaicSet->AddYCrCbImage((unsigned char*)(void*)pBuffer, layerIndex, cameraIndex, triggerIndex);
			}

			void ClearAllImages()
			{
				_pMosaicSet->ClearAllImages();
			}

			void SetThreadNumber(unsigned int iValue)
			{
				_pMosaicSet->SetThreadNumber((int)iValue);
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
			double GetNominalPixelSizeX(){return _pMosaicSet->GetNominalPixelSizeX();}
			double GetNominalPixelSizeY(){return _pMosaicSet->GetNominalPixelSizeY();}
			unsigned int GetObjectWidthInPixels(){return _pMosaicSet->GetObjectWidthInPixels();}
			unsigned int GetObjectLengthInPixels(){return _pMosaicSet->GetObjectLengthInPixels();}
			unsigned int GetImageWidthInPixels(){return _pMosaicSet->GetImageWidthInPixels();}
			unsigned int GetImageLengthInPixels(){return _pMosaicSet->GetImageHeightInPixels();}

			/// Saves all stitched images to a folder...
			bool SaveAllStitchedImagesToDirectory(System::String^ directoryName);	

			/// Loads all stitched images from a folder...
			bool LoadAllStitchedImagesFromDirectory(System::String^ dirrectoryName);

			///
			/// Copy transforms from a different mosaic set...
			///	This is currently needed for sentry because the panel aligner only works with a single panel.
			///
			bool CopyTransforms(ManagedMosaicSet ^pSet);

			///
			/// Copy buffers from a different mosaic set...
			///	This is currently needed for sentry because the panel aligner only works with a single panel.
			///
			bool CopyBuffers(ManagedMosaicSet ^pSet);

			void SetSeperateProcessStages(bool bValue) 
			{
				_pMosaicSet->SetSeperateProcessStages(bValue);
			}

			bool SetFiducailCadLoc(int iID, double dx, double dy)
			{
				return(_pMosaicSet->SetFiducailCadLoc(iID, dx, dy));
			}

			bool SetFiducialFovLoc(int iID, 
				int iLayer, int iTrig, int iCam,
				double dCol, double dRow)
			{
				return(_pMosaicSet->SetFiducialFovLoc(
					iID, iLayer, iTrig, iCam, dCol, dRow));
			}

			void SetGaussianDemosaic(bool bValue)
			{
				_pMosaicSet->SetGaussianDemosaic(bValue);
			}

		private:
			MosaicDM::MosaicSet *_pMosaicSet;
		
			ImageAddedDelegate ^_imageAddedDelegate;
			void RaiseImageAdded(unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex)
			{
				OnImageAdded(layerIndex, cameraIndex, triggerIndex);
			}
	};
}
