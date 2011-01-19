#include "StdAfx.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "MosaicTile.h"

namespace MosaicDM 
{
	MosaicLayer::MosaicLayer()
	{
		_pMosaicSet = NULL;
		_pTileArray = NULL;
		_maskImages = NULL;
		_numCameras = 0;
		_numTriggers = 0;
		_bAlignWithCAD = false;
		_bAlignWithFiducial = true;
		_bIsMaskImgValid = false;
	}

	MosaicLayer::~MosaicLayer(void)
	{
		if(_pTileArray != NULL) 
			delete[] _pTileArray;
		if(_maskImages != NULL) 
			delete [] _maskImages;
	}

	unsigned int MosaicLayer::Index()
	{
		return _layerIndex;
	}

	void MosaicLayer::Initialize(MosaicSet *pMosaicSet, 
        							int numCameras,
									int numTriggers,
									bool bAlignWithCAD,
									bool bAlignWithFiducial,
									unsigned int layerIndex)
	{
		_pMosaicSet = pMosaicSet;
		_numCameras = numCameras;
		_numTriggers = numTriggers;
		_bAlignWithCAD = bAlignWithCAD;
		_bAlignWithFiducial = bAlignWithFiducial;
		_layerIndex = layerIndex;

		int numTiles = GetNumberOfTiles();
		_pTileArray = new MosaicTile[numTiles];
		_maskImages = new Image[numTiles];

		for(int i=0; i<numTiles; i++)
		{
			// @todo - set X/Y center offsets nominally...
			_pTileArray[i].Initialize(this, 0.0, 0.0);

			_maskImages[i].Configure(_pMosaicSet->GetImageWidthInPixels(), 
				_pMosaicSet->GetImageHeightInPixels(), _pMosaicSet->GetImageStrideInPixels(), 
				false);

		}
	}

	MosaicTile* MosaicLayer::GetTile(int cameraIndex, int triggerIndex)
	{
		if(cameraIndex<0 || cameraIndex>=GetNumberOfCameras() || triggerIndex<0 || triggerIndex>=GetNumberOfTriggers())
			return NULL;

		return &_pTileArray[cameraIndex*GetNumberOfTriggers()+triggerIndex];
	}

	Image* MosaicLayer::GetImage(int cameraIndex, int triggerIndex)
	{
		MosaicTile* pTile = GetTile(cameraIndex, triggerIndex);
		if(pTile == NULL)
			return NULL;

		return (Image*)pTile;
	}

	int MosaicLayer::GetNumberOfTiles()
	{
		return GetNumberOfTriggers()*GetNumberOfCameras();
	}
	
	bool MosaicLayer::HasAllImages()
	{
		int numTiles = GetNumberOfTiles();
		for(int i=0; i<numTiles; i++)
			if(!_pTileArray[i].ContainsImage())
				return false;

		return true;
	}

	void MosaicLayer::ClearAllImages()
	{
		int numTiles = GetNumberOfTiles();
		for(int i=0; i<numTiles; i++)
		{
			_pTileArray[i].ClearImageBuffer();
			_pTileArray[i].SetTransform(_pTileArray[i].GetNominalTransform());
			_maskImages[i].SetTransform(_maskImages[i].GetNominalTransform());
		}	
		_bIsMaskImgValid = false;
	}

	bool MosaicLayer::AddImage(unsigned char *pBuffer, int cameraIndex, int triggerIndex)
	{
		MosaicTile* pTile = GetTile(cameraIndex, triggerIndex);
		if(pTile == NULL)
			return false;

		return pTile->SetImageBuffer(pBuffer);
	}

	// Camera centers in Y of world space
	void MosaicLayer::CameraCentersInY(double* pdCenY)
	{
		for(int iCam=0; iCam<GetNumberOfCameras(); iCam++)
		{
			pdCenY[iCam] = 0;
			for(unsigned int iTrig=0; iTrig<GetNumberOfTriggers(); iTrig++)
			{
				pdCenY[iCam] += GetTile(iCam, iTrig)->CenterY();
			}
			pdCenY[iCam] /= GetNumberOfTriggers();
		}
	}

	// Trigger centesr in  X of world space 
	void MosaicLayer::TriggerCentersInX(double* pdCenX)
	{
		for(unsigned int iTrig=0; iTrig<GetNumberOfTriggers(); iTrig++)
		{
			pdCenX[iTrig] = 0;
			for(unsigned int iCam=0; iCam<GetNumberOfCameras(); iCam++)
			{
				pdCenX[iTrig] += GetTile(iCam, iTrig)->CenterX();
			}
			pdCenX[iTrig] /= GetNumberOfCameras();
		}

	}

	// Prepare Mask images to use (validate mask images)
	bool MosaicLayer::PrepareMaskImages()
	{
		// Validation check
		if(!HasAllImages()) return(false);

		for(unsigned int i=0 ; i<GetNumberOfTiles(); i++)
		{
			_maskImages[i].SetTransform(_pTileArray[i].GetTransform());
			_maskImages[i].CreateOwnBuffer();
		}

		_bIsMaskImgValid = true;

		return true;
	}

	// Get a mask image point in certain position
	// return NULL if it is not valid
	Image* MosaicLayer::GetMaskImage(unsigned int iCamIndex, unsigned int iTrigIndex) 
	{
		// Validation check
		if(!_bIsMaskImgValid)
			return NULL;

		unsigned int iPos = iTrigIndex* GetNumberOfCameras() + iCamIndex;
		return(&_maskImages[iPos]);
	}
}