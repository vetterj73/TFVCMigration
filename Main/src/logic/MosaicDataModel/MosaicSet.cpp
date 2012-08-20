// This is the main DLL file.

#include "stdafx.h"
#include "MosaicSet.h"
#include "MosaicLayer.h"
#include "CorrelationFlags.h"
#include "Utilities.h"
#include "Bitmap.h"
#include "ColorImage.h"
#include "DemosaicJob.h"
#include <ctime>

namespace MosaicDM 
{

#pragma region class FiducialLocation
	FiducialLocation::FiducialLocation()
	{
		dCadX = -1;
		dCadY = -1;

		Reset();
	}

	FiducialLocation::FiducialLocation(double cadX, double cadY)
	{
		dCadX = cadX;
		dCadY = cadY;

		Reset();
	}

	void FiducialLocation::Reset()
	{
		iLayerIndex = -1;
		iTrigIndex = -1;
		iCamIndex = -1;
		dCol = -1;
		dRow = -1;
	}

	bool FiducialLocation::IsValid()
	{
		if(iLayerIndex < 0 ||
			iTrigIndex < 0 ||
			iCamIndex < 0 ||
			dCol < 0 ||
			dRow < 0)
			return(false);

		return(true);
	}
#pragma endregion

	MosaicSet::MosaicSet(double objectWidthInMeters,
					  double objectLengthInMeters,
					  unsigned int imageWidthInPixels,
					  unsigned int imageHeightInPixels,
					  unsigned int imageStrideInPixels,
					  double nominalPixelSizeXInMeters,
					  double nominalPixelSizeYInMeters,
					  bool ownBuffers,
					  bool bBayerPattern,
					  int iBayerType,
					  bool bSkipDemosaic)
	{
		_objectWidthInMeters = objectWidthInMeters;
		_objectLengthInMeters = objectLengthInMeters;
		_imageWidth = imageWidthInPixels;
		_imageHeight = imageHeightInPixels;
		_imageStride = imageStrideInPixels;
		_pixelSizeX = nominalPixelSizeXInMeters;
		_pixelSizeY = nominalPixelSizeYInMeters;
		_registeredImageAddedCallback = NULL;
		_pCallbackContext = NULL;
		_ownBuffers = ownBuffers;
		
		_bBayerPattern = bBayerPattern;
		_iBayerType = iBayerType;
		_bSkipDemosaic = bSkipDemosaic;

		_pDemosaicJobManager = NULL;
		_demosaicJobPtrList.clear();
		_iNumThreads = 8;

		_bSeperateProcessStages = false;

		_inputFidLocMap.clear();
	}

	MosaicSet::~MosaicSet()
	{
		for(unsigned int i=0; i<_layerList.size(); i++)
		{
			for(unsigned int j=i; j<_layerList.size(); j++)
			{
				pair<unsigned int, unsigned int> Pair(i, j);
				delete _correlationFlagsMap[Pair];
			}

			delete _layerList[i];
		}	
		
		_layerList.clear();
		_correlationFlagsMap.clear();

		if(_pDemosaicJobManager != NULL)
			delete _pDemosaicJobManager;
	}

	bool MosaicSet::CopyTransforms(MosaicSet *pMosaicSet)
	{
		// Sanity Check that we have the same size mosaic
		if(GetNumMosaicLayers() != pMosaicSet->GetNumMosaicLayers())
			return false;

		// Copy transform from each image in each layer...
		for(unsigned int i=0; i<_layerList.size(); i++)
		{
			if(GetLayer(i)->GetNumberOfCameras() != pMosaicSet->GetLayer(i)->GetNumberOfCameras() ||
				GetLayer(i)->GetNumberOfTriggers() != pMosaicSet->GetLayer(i)->GetNumberOfTriggers())
				return false;

			
			for(unsigned int iTrig=0; iTrig<GetLayer(i)->GetNumberOfTriggers(); iTrig++)
				for(unsigned int iCam=0; iCam<GetLayer(i)->GetNumberOfCameras(); iCam++)
					GetLayer(i)->GetImage(iTrig, iCam)->SetTransform(pMosaicSet->GetLayer(i)->GetImage(iTrig,iCam)->GetTransform());
		}	

		return true;
	}

	bool MosaicSet::CopyBuffers(MosaicSet *pMosaicSet)
	{
		// Sanity Check that we have the same size mosaic
		if(GetNumMosaicLayers() != pMosaicSet->GetNumMosaicLayers())
			return false;

		// Copy transform from each image in each layer...
		for(unsigned int i=0; i<_layerList.size(); i++)
		{
			if(GetLayer(i)->GetNumberOfCameras() != pMosaicSet->GetLayer(i)->GetNumberOfCameras() ||
				GetLayer(i)->GetNumberOfTriggers() != pMosaicSet->GetLayer(i)->GetNumberOfTriggers())
				return false;

			for(unsigned int iTrig=0; iTrig<GetLayer(i)->GetNumberOfTriggers(); iTrig++)
				for(unsigned int iCam=0; iCam<GetLayer(i)->GetNumberOfCameras(); iCam++)
					GetLayer(i)->GetImage(iTrig, iCam)->SetBuffer(pMosaicSet->GetLayer(i)->GetImage(iTrig, iCam)->GetBuffer());
		}	

		return true;
	}

	MosaicLayer *MosaicSet::GetLayer(unsigned int index)
	{
		if(index<0 || index >= _layerList.size())
			return NULL;

		return _layerList[index];
	}

	MosaicLayer * MosaicSet::AddLayer(
		unsigned int numCameras,
		unsigned int numTriggers,
		bool bAlignWithCAD,
		bool bAlignWithFiducial,
		bool bFiducialBrighterThanBackground,
		bool bFiducialAllowNegativeMatch,
		unsigned int iDeviceIndex)
	{
		FireLogEntry(LogTypeDiagnostic, "Layer Added to Mosaic!");

		MosaicLayer *pML = new MosaicLayer();

		pML->Initialize(this, numCameras, numTriggers, bAlignWithCAD, 
			bAlignWithFiducial, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch,
			(unsigned int)_layerList.size(), iDeviceIndex);
		_layerList.push_back(pML);

		// Setup the default correlation Flags...
		for(unsigned int i=0; i<_layerList.size(); i++)
		{
			CorrelationFlags *pFlags = new CorrelationFlags();	
			pair<int, int> Pair(i, _layerList.size()-1);
			_correlationFlagsMap[Pair] = pFlags;
		}

		return pML;
	}

	unsigned int MosaicSet::GetMosaicTotalNumberOfTriggers()
	{
		// total up all of the triggers in each layer
		unsigned int totalTrigs(0);
		for(unsigned int i=0; i<_layerList.size(); i++)
			totalTrigs += GetLayer(i)->GetNumberOfTriggers();
		return totalTrigs;
	}


	CorrelationFlags* MosaicSet::GetCorrelationFlags(unsigned int layerX, unsigned int layerY)
	{
		if(layerX > _layerList.size()-1 || layerY > _layerList.size()-1 ||
			layerX<0 || layerY<0)
			return NULL;

		if(layerX > layerY)
		{
			int temp = layerX;
			layerX = layerY;
			layerY = temp;
		}

		pair<unsigned int, unsigned int> Pair(layerX, layerY);

		return _correlationFlagsMap[Pair];
	}

	unsigned int MosaicSet::GetNumDevice()
	{
		list<unsigned int> deviceIndice;
		deviceIndice.clear();

		for(LayerListIterator i=_layerList.begin(); i!=_layerList.end(); i++)
		{
			unsigned int iIndex = (*i)->DeviceIndex();

			// Whether index is contained
			bool bContain = false;
			for(list<unsigned int>::iterator j=deviceIndice.begin(); j!=deviceIndice.end(); j++)
			{
				if((*j) == iIndex)
				{
					bContain = true;
					break;
				}
			}

			// If not contained, add it
			if(!bContain)
				deviceIndice.push_back(iIndex);
		}

		return(deviceIndice.size());
	}

	bool MosaicSet::HasAllImages()
	{
		for(unsigned int i=0; i<_layerList.size(); i++)
			if(!_layerList[i]->HasAllImages())
				return false;

		return true;
	}

	void MosaicSet::ClearAllImages()
	{
		for(unsigned int i=0; i<_layerList.size(); i++)
			_layerList[i]->ClearAllImages();

		// Reset input fiducial Fov location.
		if(HasInputFidLocations())
			ResetInputFidLocMap();
	}

	int MosaicSet::NumberOfImageTiles()
	{
		int iNum = 0;
		for(unsigned int i=0; i<_layerList.size(); i++)
			iNum += _layerList[i]->GetNumberOfTiles();

		return(iNum);
	}

	bool MosaicSet::SaveAllStitchedImagesToDirectory(string directoryName)
	{
		string fullDir = directoryName;
		//if(directoryName[directoryName.size()-1] != '/' &&
		//   directoryName[directoryName.size()-1] != '\\')
		//   fullDir += "\\";

		for(unsigned int i=0; i<GetNumMosaicLayers(); i++)
		{
			MosaicLayer *pLayer = GetLayer(i);
			Image *pImage = pLayer->GetStitchedImage();
			if(pImage == NULL)
				return false;

			char buffer[20];
			sprintf(buffer, "layer%d.bmp", i);
			string file = fullDir + buffer;

			if(_bBayerPattern)
				((ColorImage*)pImage)->Save(file);
			else
				pImage->Save(file);
		}

		return true;
	}

	bool MosaicSet::LoadAllStitchedImagesFromDirectory(string directoryName)
	{
		string fullDir = directoryName;
		if(directoryName[directoryName.size()-1] != '/' &&
		   directoryName[directoryName.size()-1] != '\\')
		   fullDir += "\\";

		// if we are loading from disk, remove all of the existing raw images...
		ClearAllImages();
		
		for(unsigned int i=0; i<GetNumMosaicLayers(); i++)
		{
			MosaicLayer *pLayer = GetLayer(i);

			char buffer[20];
			sprintf(buffer, "layer%d.bmp", i);
			string file = fullDir + buffer;
			
			Bitmap bmp;
			bmp.read(file);
			if(bmp.width() != GetObjectLengthInPixels() ||
			   bmp.height() != GetObjectWidthInPixels())
				return false;

			// Check gray/color match
			if(IsBayerPattern() && !IsSkipDemosaic())
			{
				if(bmp.bytesPerPixel() != 3)
					return(false);
			}
			else
			{		
				if(bmp.bytesPerPixel() != 1)
					return(false);
			}

			pLayer->SetStitchedBuffer(bmp.GetBuffer());
		}

		return true;
	}

	void MosaicSet::RegisterImageAddedCallback(IMAGEADDED_CALLBACK pCallback, void* pContext)
	{
		_registeredImageAddedCallback = pCallback;
		_pCallbackContext = pContext;
	}

	void MosaicSet::UnregisterImageAddedCallback()
	{
		_registeredImageAddedCallback = NULL;
		_pCallbackContext = NULL;
	}

	// Input buffer need to be Bayer or grayscale
	bool MosaicSet::AddRawImage(unsigned char *pBuffer, unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex)
	{
		if(!_bSeperateProcessStages)	// Normal working mode
		{
			// If bayer pattern and demosaic is needed.
			// Work in Multi-thread to speed up
			if(_bBayerPattern && !_bSkipDemosaic)
			{
				// Create the thread job manager if it is necessary
				if(_pDemosaicJobManager == NULL)
					_pDemosaicJobManager = new CyberJob::JobManager("Demosaic", _iNumThreads);
		
				// Add demosaic job to thread manager
				DemosaicJob* pJob = new DemosaicJob(this, pBuffer, layerIndex, cameraIndex, triggerIndex);
				_pDemosaicJobManager->AddAJob((CyberJob::Job*)pJob);
				_demosaicJobPtrList.push_back(pJob);

				// If all images are added, clean up
				if(_demosaicJobPtrList.size() == NumberOfImageTiles())
				{
					// Wait all demosaics are done
					_pDemosaicJobManager->MarkAsFinished();
					while(_pDemosaicJobManager->TotalJobs() > 0)
						Sleep(10);

					// Clear job list
					list<DemosaicJob*>::iterator i;
					for(i = _demosaicJobPtrList.begin(); i!= _demosaicJobPtrList.end(); i++)
						delete (*i);

					_demosaicJobPtrList.clear();
				
					FireLogEntry(LogTypeDiagnostic, "Demosaic is done!");
				}
			}
			// If greyscale or demosaic is not needed. 
			// Not necessary to add overheader by use multi-thread manager 
			else	
			{
				MosaicLayer *pLayer = GetLayer(layerIndex);
				if(pLayer == NULL)
					return false;

				if(!pLayer->AddRawImage(pBuffer, cameraIndex, triggerIndex))
					return false;

				FireImageAdded(layerIndex, cameraIndex, triggerIndex);
			}
		}
		else // Speed test mode only
		{
			// Add acquired FOV data into list
			FovData fovData;
			fovData.pFovRawData = pBuffer;
			fovData.iLayerIndex = layerIndex;
			fovData.iTrigIndex = triggerIndex;
			fovData.iCamIndex = cameraIndex;
			_fovDataList.push_back(fovData);

			// If all Fovs are collected
			if(_fovDataList.size() == NumberOfImageTiles())
			{
				FireLogEntry(LogTypeDiagnostic, "End SIM1 acquisition");
				// If bayer pattern and demosaic is needed.
				// Work in Multi-thread to speed up
				if(_bBayerPattern && !_bSkipDemosaic)
				{
					FireLogEntry(LogTypeDiagnostic, "Begin demosaic");
					clock_t StartTime = clock();

					// Create the thread job manager if it is necessary
					if(_pDemosaicJobManager == NULL)
						_pDemosaicJobManager = new CyberJob::JobManager("Demosaic", _iNumThreads);

					for(list<FovData>::iterator i = _fovDataList.begin(); i != _fovDataList.end(); i++)
					{
						// Add demosaic job to thread manager
						// Not send event to aligner
						DemosaicJob* pJob = new DemosaicJob(this, i->pFovRawData, i->iLayerIndex, i->iCamIndex, i->iTrigIndex, false);
						_pDemosaicJobManager->AddAJob((CyberJob::Job*)pJob);
						_demosaicJobPtrList.push_back(pJob);
					}

					// Wait all demosaics are done
					_pDemosaicJobManager->MarkAsFinished();
					while(_pDemosaicJobManager->TotalJobs() > 0)
						Sleep(10);

					// Clear job list
					list<DemosaicJob*>::iterator i;
					for(i = _demosaicJobPtrList.begin(); i!= _demosaicJobPtrList.end(); i++)
						delete (*i);

					_demosaicJobPtrList.clear();

					FireLogEntry(LogTypeDiagnostic, "End demosaic, Time = %f", (float)(clock() - StartTime)/CLOCKS_PER_SEC);
				}
				else
				{	
					// If greyscale or demosaic is not needed. 
					// Not necessary to add overheader by use multi-thread manager 
					for(list<FovData>::iterator i = _fovDataList.begin(); i != _fovDataList.end(); i++)
					{
						MosaicLayer *pLayer = GetLayer(i->iLayerIndex);
						if(pLayer == NULL)
							return false;

						if(!pLayer->AddRawImage(i->pFovRawData, i->iCamIndex, i->iTrigIndex))
							return false;
					}
				}

				// Send events to aligner
				FireLogEntry(LogTypeDiagnostic, "Begin cyberstitch alignment");
				for(list<FovData>::iterator i = _fovDataList.begin(); i != _fovDataList.end(); i++)
				{
					FireImageAdded(i->iLayerIndex, i->iCamIndex, i->iTrigIndex);
				}
				// Clear fovdata list for next panel
				_fovDataList.clear();
			}
		}

		return true;
	}

	// Input buffer need to be YCrCb
	bool MosaicSet::AddYCrCbImage(unsigned char *pBuffer, unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex)
	{
		MosaicLayer *pLayer = GetLayer(layerIndex);
		if(pLayer == NULL)
			return false;

		if(!pLayer->AddYCrCbImage(pBuffer, cameraIndex, triggerIndex))
			return false;

		FireImageAdded(layerIndex, cameraIndex, triggerIndex);

		return true;
	}

	void MosaicSet::FireImageAdded(unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex)
	{
		if(_registeredImageAddedCallback != NULL)
			_registeredImageAddedCallback(layerIndex, cameraIndex, triggerIndex, _pCallbackContext);
	}

	unsigned int MosaicSet::GetObjectWidthInPixels()
	{
		return GetNumPixels(GetObjectWidthInMeters(), GetNominalPixelSizeX());
	}
	
	unsigned int MosaicSet::GetObjectLengthInPixels()
	{
		return GetNumPixels(GetObjectLengthInMeters(), GetNominalPixelSizeY());
	}

	// For fiducial CAD information from outside
	bool MosaicSet::SetFiducailCadLoc(int iID, double dx, double dy)
	{
		// If already exists, return false
		if(_inputFidLocMap.find(iID) != _inputFidLocMap.end())
			return(false);

		// Add fiducial CAD information
		FiducialLocation fidLoc(dx, dy);
		_inputFidLocMap.insert(pair<int, FiducialLocation>(iID, fidLoc));

		return(true);
	}

	bool MosaicSet::SetFiducialFovLoc(int iID, 
			int iLayer, int iTrig, int iCam,
			double dCol, double dRow)
	{
		// If doesn't exist, return false
		if(_inputFidLocMap.find(iID) == _inputFidLocMap.end())
			return(false);

		_inputFidLocMap[iID].iLayerIndex = iLayer;
		_inputFidLocMap[iID].iTrigIndex = iTrig;
		_inputFidLocMap[iID].iCamIndex = iCam;
		_inputFidLocMap[iID].dCol = dCol;
		_inputFidLocMap[iID].dRow = dRow;
	}

	bool MosaicSet::HasInputFidLocations()
	{
		return(!_inputFidLocMap.empty());
	}

	bool MosaicSet::IsValidInputFidLocations()
	{
		for(map<int, FiducialLocation>::iterator i=_inputFidLocMap.begin(); i!=_inputFidLocMap.end(); i++)
		{
			if(!i->second.IsValid())
				return(false);
		}

		return(true);
	}

	void MosaicSet::ResetInputFidLocMap()
	{
		for(map<int, FiducialLocation>::iterator i=_inputFidLocMap.begin(); i!=_inputFidLocMap.end(); i++)
		{
			i->second.Reset();
		}
	}

	map<int, FiducialLocation>* MosaicSet::GetInputFidLocMap()
	{
		return(&_inputFidLocMap);
	}
}