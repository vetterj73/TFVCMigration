#include "StdAfx.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "MosaicTile.h"
#include "MorphJob.h"
#include "JobManager.h"

using namespace CyberJob;
namespace MosaicDM 
{
	MosaicLayer::MosaicLayer()
	{
		_pMosaicSet = NULL;
		_pTileArray = NULL;
		_maskImages = NULL;
		_pStitchedImage = NULL;
		_numCameras = 0;
		_numTriggers = 0;
		_bAlignWithCAD = false;
		_bAlignWithFiducial = true;
		_bIsMaskImgValid = false;
		_stitchedImageValid = false;
	}

	MosaicLayer::~MosaicLayer(void)
	{
		if(_pTileArray != NULL) 
			delete[] _pTileArray;
		if(_maskImages != NULL) 
			delete [] _maskImages;
		if(_pStitchedImage != NULL)
			delete _pStitchedImage;
	}

	unsigned int MosaicLayer::Index()
	{
		return _layerIndex;
	}

	void MosaicLayer::Initialize(MosaicSet *pMosaicSet, 
        							unsigned int numCameras,
									unsigned int numTriggers,
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

		unsigned int numTiles = GetNumberOfTiles();
		_pTileArray = new MosaicTile[numTiles];
		_maskImages = new Image[numTiles];

		for(unsigned int i=0; i<numTiles; i++)
		{
			// @todo - set X/Y center offsets nominally...
			_pTileArray[i].Initialize(this, 0.0, 0.0);

			_maskImages[i].Configure(_pMosaicSet->GetImageWidthInPixels(), 
				_pMosaicSet->GetImageHeightInPixels(), _pMosaicSet->GetImageStrideInPixels(), 
				false);

		}
	}

	void MosaicLayer::SetStitchedBuffer(unsigned char *pBuffer)
	{
		AllocateStitchedImageIfNecessary();
		memcpy(_pStitchedImage->GetBuffer(), pBuffer, _pMosaicSet->GetObjectWidthInPixels()*_pMosaicSet->GetObjectLengthInPixels());
		_stitchedImageValid = true;
	}

	Image *MosaicLayer::GetStitchedImage()
	{
		CreateStitchedImageIfNecessary();

		return _pStitchedImage;
	}

	void MosaicLayer::AllocateStitchedImageIfNecessary()
	{
		// Create it once for each panel type...
		if(_pStitchedImage != NULL)
			return;

		_pStitchedImage = new Image();
		ImgTransform inputTransform;
		inputTransform.Config(_pMosaicSet->GetNominalPixelSizeX(), 
			_pMosaicSet->GetNominalPixelSizeY(), 0, 0, 0);
			
		unsigned int iNumRows = _pMosaicSet->GetObjectWidthInPixels();
		unsigned int iNumCols = _pMosaicSet->GetObjectLengthInPixels();

		_pStitchedImage->Configure(iNumCols, iNumRows, iNumCols, inputTransform, inputTransform, true);
	}

	// @todo - Need to redo this for each new panel!!!!
	void MosaicLayer::CreateStitchedImageIfNecessary()
	{
		if(_stitchedImageValid)
			return;

		_stitchedImageValid = true;
		AllocateStitchedImageIfNecessary();

		// Trigger and camera centers in world space
		unsigned int iNumTrigs = GetNumberOfTriggers();
		unsigned int iNumCams = GetNumberOfCameras();
		double* pdCenX = new double[iNumTrigs];
		double* pdCenY = new double[iNumCams];
		TriggerCentersInX(pdCenX);
		CameraCentersInY(pdCenY);

		// Panel image Row bounds for Roi (decreasing order)
		int* piRectRows = new int[iNumTrigs+1];
		piRectRows[0] = _pStitchedImage->Rows();
		for(unsigned int i=1; i<iNumTrigs; i++)
		{
			double dX = (pdCenX[i-1] +pdCenX[i])/2;
			double dTempRow, dTempCol;
			_pStitchedImage->WorldToImage(dX, 0, &dTempRow, &dTempCol);
			piRectRows[i] = (int)dTempRow;
			if(piRectRows[i]>=(int)_pStitchedImage->Rows()) piRectRows[i] = _pStitchedImage->Rows();
			if(piRectRows[i]<0) piRectRows[i] = 0;
		}
		piRectRows[iNumTrigs] = 0; 

		// Panel image Column bounds for Roi (increasing order)
		int* piRectCols = new int[iNumCams+1];
		piRectCols[0] = 0;
		for(unsigned int i=1; i<iNumCams; i++)
		{
			double dY = (pdCenY[i-1] +pdCenY[i])/2;
			double dTempRow, dTempCol;
			_pStitchedImage->WorldToImage(0, dY, &dTempRow, &dTempCol);
			piRectCols[i] = (int)dTempCol;
			if(piRectCols[i]<0) piRectCols[i] = 0;
			if(piRectCols[i]>(int)_pStitchedImage->Columns()) piRectCols[i] = _pStitchedImage->Columns();;
		}
		piRectCols[iNumCams] = _pStitchedImage->Columns();

		char buf[20];
		sprintf_s(buf, 19, "Stitcher%d", _layerIndex);
		CyberJob::JobManager jm(buf, 1);
		vector<MorphJob*> morphJobs;
		// Morph each Fov to create stitched panel image
		for(unsigned int iTrig=0; iTrig<iNumTrigs; iTrig++)
		{
			for(unsigned int iCam=0; iCam<iNumCams; iCam++)
			{
				Image* pFOV = GetImage(iCam, iTrig);

				MorphJob *pJob = new MorphJob(_pStitchedImage, pFOV,
					(unsigned int)piRectCols[iCam], (unsigned int)piRectRows[iTrig+1], 
					(unsigned int)(piRectCols[iCam+1]-1), (unsigned int)(piRectRows[iTrig]-1));
				jm.AddAJob((Job*)pJob);
				morphJobs.push_back(pJob);
			}
		}

		// Wait until it is complete...
		jm.MarkAsFinished();
		while(jm.TotalJobs() > 0)
			Sleep(10);

		for(unsigned int i=0; i<morphJobs.size(); i++)
			delete morphJobs[i];
		morphJobs.clear();
		delete [] pdCenX;
		delete [] pdCenY;
		delete [] piRectRows;
		delete [] piRectCols;
	}

	MosaicTile* MosaicLayer::GetTile(unsigned int cameraIndex, unsigned int triggerIndex)
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

	unsigned int MosaicLayer::GetNumberOfTiles()
	{
		return GetNumberOfTriggers()*GetNumberOfCameras();
	}
	
	bool MosaicLayer::HasAllImages()
	{
		unsigned int numTiles = GetNumberOfTiles();
		for(unsigned int i=0; i<numTiles; i++)
			if(!_pTileArray[i].ContainsImage())
				return false;

		return true;
	}

	void MosaicLayer::ClearAllImages()
	{
		unsigned int numTiles = GetNumberOfTiles();
		for(unsigned int i=0; i<numTiles; i++)
		{
			_pTileArray[i].ClearImageBuffer();
			_pTileArray[i].SetTransform(_pTileArray[i].GetNominalTransform());
			_maskImages[i].SetTransform(_maskImages[i].GetNominalTransform());
		}	
		_bIsMaskImgValid = false;
		_stitchedImageValid = false;
	}

	bool MosaicLayer::AddImage(unsigned char *pBuffer, unsigned int cameraIndex, unsigned int triggerIndex)
	{
		MosaicTile* pTile = GetTile(cameraIndex, triggerIndex);
		if(pTile == NULL)
			return false;

		return pTile->SetImageBuffer(pBuffer);
	}

	// Camera centers in Y of world space
	void MosaicLayer::CameraCentersInY(double* pdCenY)
	{
		for(unsigned int iCam=0; iCam<GetNumberOfCameras(); iCam++)
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
		//if(!HasAllImages()) return(false);

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
		//if(!_bIsMaskImgValid)
			//return NULL;

		unsigned int iPos = iTrigIndex* GetNumberOfCameras() + iCamIndex;
		return(&_maskImages[iPos]);
	}
}