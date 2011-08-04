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

		_piStitchGridRows = NULL;
		_piStitchGridCols = NULL;
	}

	MosaicLayer::~MosaicLayer(void)
	{
		if(_pTileArray != NULL) 
			delete[] _pTileArray;
		if(_maskImages != NULL) 
			delete [] _maskImages;
		if(_pStitchedImage != NULL)
			delete _pStitchedImage;
		
		if(_piStitchGridRows != NULL)
			delete [] _piStitchGridRows;

		if(_piStitchGridCols != NULL)
			delete [] _piStitchGridCols;
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

		_piStitchGridRows = new int[_numTriggers+1];
		_piStitchGridCols = new int[_numCameras+1]; 

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

	Image *MosaicLayer::GetStitchedImage(bool bRecreate)
	{
		if(bRecreate) 
			_stitchedImageValid = false;

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

	// Calculate grid for stitching image
	bool MosaicLayer::CalculateStitchGrids()
	{
		if(_pStitchedImage == NULL)
			return(false);
		
		// Trigger and camera centers in world space
		unsigned int iNumTrigs = GetNumberOfTriggers();
		unsigned int iNumCams = GetNumberOfCameras();
		double* pdCenX = new double[iNumTrigs];
		double* pdCenY = new double[iNumCams];
		TriggerCentersInX(pdCenX);
		CameraCentersInY(pdCenY);

		// Panel image Row bounds for Roi (decreasing order)
		_piStitchGridRows[0] = _pStitchedImage->Rows();
		for(unsigned int i=1; i<iNumTrigs; i++)
		{
			double dX = (pdCenX[i-1] +pdCenX[i])/2;
			double dTempRow, dTempCol;
			_pStitchedImage->WorldToImage(dX, 0, &dTempRow, &dTempCol);
			_piStitchGridRows[i] = (int)dTempRow;
			if(_piStitchGridRows[i]>=(int)_pStitchedImage->Rows()) _piStitchGridRows[i] = _pStitchedImage->Rows();
			if(_piStitchGridRows[i]<0) _piStitchGridRows[i] = 0;
		}
		_piStitchGridRows[iNumTrigs] = 0; 

		// Panel image Column bounds for Roi (increasing order)
		_piStitchGridCols[0] = 0;
		for(unsigned int i=1; i<iNumCams; i++)
		{
			double dY = (pdCenY[i-1] +pdCenY[i])/2;
			double dTempRow, dTempCol;
			_pStitchedImage->WorldToImage(0, dY, &dTempRow, &dTempCol);
			_piStitchGridCols[i] = (int)dTempCol;
			if(_piStitchGridCols[i]<0) _piStitchGridCols[i] = 0;
			if(_piStitchGridCols[i]>(int)_pStitchedImage->Columns()) _piStitchGridCols[i] = _pStitchedImage->Columns();;
		}
		_piStitchGridCols[iNumCams] = _pStitchedImage->Columns();

		delete [] pdCenX;
		delete [] pdCenY;

		return(true);
	}

	// @todo - Need to redo this for each new panel!!!!
	void MosaicLayer::CreateStitchedImageIfNecessary()
	{
		if(_stitchedImageValid)
			return;

		_stitchedImageValid = true;
		AllocateStitchedImageIfNecessary();

		// Calcaute the grid for stitching image
		if(!CalculateStitchGrids())
			return;
		
		_pMosaicSet->FireLogEntry(LogTypeDiagnostic, "Layer#%d: Begin creating stitched image", _layerIndex); 
		
		unsigned int iNumTrigs = GetNumberOfTriggers();
		unsigned int iNumCams = GetNumberOfCameras();
		char buf[20];
		sprintf_s(buf, 19, "Stitcher%d", _layerIndex);
		CyberJob::JobManager jm(buf, 8);
		vector<MorphJob*> morphJobs;
		// Morph each Fov to create stitched panel image
		for(unsigned int iTrig=0; iTrig<iNumTrigs; iTrig++)
		{
			for(unsigned int iCam=0; iCam<iNumCams; iCam++)
			{
				Image* pFOV = GetImage(iCam, iTrig);

				MorphJob *pJob = new MorphJob(_pStitchedImage, pFOV,
					(unsigned int)_piStitchGridCols[iCam], (unsigned int)_piStitchGridRows[iTrig+1], 
					(unsigned int)(_piStitchGridCols[iCam+1]-1), (unsigned int)(_piStitchGridRows[iTrig]-1));
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

		_pMosaicSet->FireLogEntry(LogTypeDiagnostic, "Layer#%d: End creating stitched image", _layerIndex); 
	}

	
	Image *MosaicLayer::GetStitchedImageWithHeight(
		unsigned char* pHeighBuf, double dHeightResolution, double dPupilDistance,
		bool bRecreate)
	{
		if(bRecreate) 
			_stitchedImageValid = false;

		CreateStitchedImageWithHeightIfNecessary(pHeighBuf, dHeightResolution, dPupilDistance);

		return _pStitchedImage;
	}

	void MosaicLayer::CreateStitchedImageWithHeightIfNecessary(
		unsigned char* pHeighBuf, double dHeightResolution, double dPupilDistance)
	{
		if(_stitchedImageValid)
			return;

		_stitchedImageValid = true;
		AllocateStitchedImageIfNecessary();		
		
		// Calcaute the grid for stitching image
		if(!CalculateStitchGrids())
			return;

		// Create height image
		Image heightImage;
		heightImage.Configure(
			_pStitchedImage->Columns(),
			_pStitchedImage->Rows(),
			_pStitchedImage->PixelRowStride(),
			_pStitchedImage->GetTransform(),
			_pStitchedImage->GetTransform(),
			false,
			pHeighBuf);

		unsigned int iNumTrigs = GetNumberOfTriggers();
		unsigned int iNumCams = GetNumberOfCameras();

		_pMosaicSet->FireLogEntry(LogTypeDiagnostic, "Layer#%d: Begin Creating stitched image with Height", _layerIndex); 
		
		char buf[20];
		sprintf_s(buf, 19, "Stitcher%d", _layerIndex);
		CyberJob::JobManager jm(buf, 8);
		vector<MorphWithHeightJob*> morphJobs;
		// Morph each Fov to create stitched panel image
		for(unsigned int iTrig=0; iTrig<iNumTrigs; iTrig++)
		{
			for(unsigned int iCam=0; iCam<iNumCams; iCam++)
			{
				Image* pFOV = GetImage(iCam, iTrig);

				MorphWithHeightJob *pJob = new MorphWithHeightJob(_pStitchedImage, pFOV,
					(unsigned int)_piStitchGridCols[iCam], (unsigned int)_piStitchGridRows[iTrig+1], 
					(unsigned int)(_piStitchGridCols[iCam+1]-1), (unsigned int)(_piStitchGridRows[iTrig]-1),
					&heightImage, dHeightResolution, dPupilDistance);
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

		_pMosaicSet->FireLogEntry(LogTypeDiagnostic, "Layer#%d: End Creating stitched image with Height", _layerIndex); 
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