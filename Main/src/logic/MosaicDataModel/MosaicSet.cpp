// This is the main DLL file.

#include "stdafx.h"
#include "MosaicSet.h"
#include "MosaicLayer.h"
#include "CorrelationFlags.h"
#include "Utilities.h"
#include "Bitmap.h"
#include "ColorImage.h"

namespace MosaicDM 
{
	MosaicSet::MosaicSet(double objectWidthInMeters,
					  double objectLengthInMeters,
					  unsigned int imageWidthInPixels,
					  unsigned int imageHeightInPixels,
					  unsigned int imageStrideInPixels,
					  double nominalPixelSizeXInMeters,
					  double nominalPixelSizeYInMeters,
					  bool ownBuffers,
					  bool bBayerPattern,
					  int iBayerType)
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
		if(_bBayerPattern)
			_ownBuffers = true;
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

			for(unsigned int j=0; j<GetLayer(i)->GetNumberOfCameras(); j++)
				for(unsigned int k=0; k<GetLayer(i)->GetNumberOfTriggers(); k++)
					GetLayer(i)->GetImage(j,k)->SetTransform(pMosaicSet->GetLayer(i)->GetImage(j,k)->GetTransform());
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

			for(unsigned int j=0; j<GetLayer(i)->GetNumberOfCameras(); j++)
				for(unsigned int k=0; k<GetLayer(i)->GetNumberOfTriggers(); k++)
					GetLayer(i)->GetImage(j,k)->SetBuffer(pMosaicSet->GetLayer(i)->GetImage(j,k)->GetBuffer());
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

	bool MosaicSet::SaveAllStitchedImagesWithHeightToDirectory(
		string directoryName,
		unsigned char* pHeighBuf, double dHeightResolution, double dPupilDistance)
	{
		string fullDir = directoryName;
		//if(directoryName[directoryName.size()-1] != '/' &&
		//   directoryName[directoryName.size()-1] != '\\')
		//   fullDir += "\\";

		for(unsigned int i=0; i<GetNumMosaicLayers(); i++)
		{
			MosaicLayer *pLayer = GetLayer(i);
			Image *pImage = pLayer->GetStitchedImage(pHeighBuf, dHeightResolution, dPupilDistance);
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

	bool MosaicSet::AddImage(unsigned char *pBuffer, unsigned int layerIndex, unsigned int cameraIndex, unsigned int triggerIndex)
	{
		MosaicLayer *pLayer = GetLayer(layerIndex);
		if(pLayer == NULL)
			return false;

		if(!pLayer->AddImage(pBuffer, cameraIndex, triggerIndex))
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
}