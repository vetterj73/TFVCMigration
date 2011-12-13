#include "StdAfx.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "MosaicTile.h"
#include "MorphJob.h"
#include "JobManager.h"
#include "ColorImage.h"

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

		_bGridBoundaryValid = false;
		_pdGridXBoundary = NULL;
		_pdGridYBoundary = NULL;

		_pHeightInfo = NULL;
		
		// For debug 
		_pGreyStitchedImage = NULL;
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

		if(_pdGridXBoundary != NULL)
			delete [] _pdGridXBoundary;

		if(_pdGridYBoundary != NULL)
			delete [] _pdGridYBoundary;

		if(_pHeightInfo != NULL)
			delete _pHeightInfo;

		// For debug
		if(_pGreyStitchedImage != NULL)
			delete _pGreyStitchedImage;
	}

	unsigned int MosaicLayer::Index()
	{
		return _layerIndex;
	}

	unsigned int MosaicLayer::DeviceIndex()
	{
		return _deviceIndex;
	}

	void MosaicLayer::Initialize(
		MosaicSet *pMosaicSet, 
        unsigned int numCameras,
		unsigned int numTriggers,
		bool bAlignWithCAD,
		bool bAlignWithFiducial,
		bool bFiducialBrighterThanBackground,
		bool bFiducialAllowNegativeMatch,
		unsigned int layerIndex,
		unsigned int deviceIndex)
	{
		_pMosaicSet = pMosaicSet;
		_numCameras = numCameras;
		_numTriggers = numTriggers;
		_bAlignWithCAD = bAlignWithCAD;
		_bAlignWithFiducial = bAlignWithFiducial;
		_layerIndex = layerIndex;
		_deviceIndex = deviceIndex;
		_bFiducialBrighterThanBackground = bFiducialBrighterThanBackground;
		_bFiducialAllowNegativeMatch = bFiducialAllowNegativeMatch;

		unsigned int numTiles = GetNumberOfTiles();
		_pTileArray = new MosaicTile[numTiles];
		_maskImages = new Image[numTiles];

		_piStitchGridRows = new int[_numTriggers+1];
		_piStitchGridCols = new int[_numCameras+1]; 

		_pdGridXBoundary = new double[_numTriggers*2];
		_pdGridYBoundary = new double[_numCameras*2];

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

	void MosaicLayer::AllocateStitchedImageIfNecessary()
	{
		// Create it once for each panel type...
		if(_pStitchedImage != NULL)
			return;

		if(GetMosaicSet()->IsBayerPattern())
			_pStitchedImage = new ColorImage(BGR, false);
		else 
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

	// pHeighBuf: input, height image buffer
	// dHeightResolution: gray level resolution of height image
	// PupilDistance: Pupil distance in meter
	// bCreate: true, force to calcaucate stitched image event it exists
	Image *MosaicLayer::GetStitchedImage(bool bRecreate)
	{
		if(bRecreate) 
			_stitchedImageValid = false;

		CreateStitchedImageIfNecessary();

		return _pStitchedImage;
	}

	void MosaicLayer::CreateStitchedImageIfNecessary()
	{
		if(_stitchedImageValid)
			return;

		_stitchedImageValid = true;
		AllocateStitchedImageIfNecessary();		
		
		// Calcaute the grid for stitching image
		if(!CalculateStitchGrids())
			return;

		// Create height image
		Image* pHeightImage = NULL;
		double dHeightResolution=0, dPupilDistance=0;
		if(_pHeightInfo != 0)
		{
			pHeightImage = new Image();
			pHeightImage->Configure(
				_pStitchedImage->Columns(),
				_pStitchedImage->Rows(),
				_pStitchedImage->PixelRowStride(),
				_pStitchedImage->GetTransform(),
				_pStitchedImage->GetTransform(),
				false,
				_pHeightInfo->pHeightBuf);

			dHeightResolution = _pHeightInfo->dHeightResolution;
			dPupilDistance = _pHeightInfo->dPupilDistance;
		}

		unsigned int iNumTrigs = GetNumberOfTriggers();
		unsigned int iNumCams = GetNumberOfCameras();

		_pMosaicSet->FireLogEntry(LogTypeDiagnostic, "Layer#%d: Begin Creating stitched image %s", _layerIndex, _pHeightInfo==NULL?"":"With Height"); 
		
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
					(unsigned int)(_piStitchGridCols[iCam+1]-1), (unsigned int)(_piStitchGridRows[iTrig]-1),
					pHeightImage, dHeightResolution, dPupilDistance);
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

		if(pHeightImage != NULL)
			delete pHeightImage;

		_pMosaicSet->FireLogEntry(LogTypeDiagnostic, "Layer#%d: End Creating stitched image %s", _layerIndex, _pHeightInfo==NULL?"":"With Height"); 
	}

	// Calculate grid boundarys
	// Fovs are organize into grid group 
	bool MosaicLayer::CalculateGridBoundary()
	{
		if(_bGridBoundaryValid) return(true);

		// Boundary for each inverse trigger (x in world)
		for(int iInvTrig=0; iInvTrig<(int)_numTriggers; iInvTrig++)
		{
			double dTopBound, dBottomBound;
			for(int iCam=0; iCam<(int)_numCameras; iCam++)
			{
				DRect rect = GetImage(iCam, _numTriggers-1-iInvTrig)->GetBoundBoxInWorld();
				if(iCam==0)
				{
					dTopBound = rect.xMin;
					dBottomBound = rect.xMax;
				}
				else
				{
					// Max of all FOV Mins
					if(dTopBound < rect.xMin) dTopBound = rect.xMin;
					// Min of all Fov Maxs
					if(dBottomBound > rect.xMax) dBottomBound = rect.xMax;
				}
			}
			// top value < bottom value
			_pdGridXBoundary[iInvTrig*2] = dTopBound;
			_pdGridXBoundary[iInvTrig*2+1] = dBottomBound;
		}

		// Boundary for each camera (y in world)
		for(int iCam=0; iCam<(int)_numCameras; iCam++)
		{
			double dLeftBound, dRightBound;
			for(int iTrig=0; iTrig<(int)_numTriggers; iTrig++)
			{
				DRect rect = GetImage(iCam, iTrig)->GetBoundBoxInWorld();
				if(iTrig==0)
				{
					dLeftBound = rect.yMin;
					dRightBound = rect.yMax;
				}
				else
				{
					// Max of all FOV Mins
					if(dLeftBound < rect.yMin) dLeftBound = rect.yMin;
					// Min of all Fov Maxs
					if(dRightBound > rect.yMax) dRightBound = rect.yMax;
				}
			}
			_pdGridYBoundary[iCam*2] = dLeftBound;
			_pdGridYBoundary[iCam*2+1] = dRightBound;
		}

		_bGridBoundaryValid = true;
		return(true);
	}

	// Get a morphed image patch
	// pBuf: inout, the buffer hold the patch, memeory is allocated before pass in
	// iPixelSpan: buffer's span in pixel
	// iStartCol, iWidth, iStartRow and iHeight: Roi in pixel
	// pPreferSelectedFov: inout, the prefer Fovs and selected Fovs 
	// pHeighImgBuf and iHeightImgSpan: input, height image buf and its span 
	// The buffer point to (0, 0) of Height image, 
	// It's origin needs not be the origin of pBuf in the world space
	// dHeightResolution: gray level resolution of height image
	// PupilDistance: Pupil distance in meter
	bool MosaicLayer::GetImagePatch(
		unsigned char* pBuf,
		unsigned int iPixelSpan,
		unsigned int iStartRowInCad,
		unsigned int iStartColInCad,
		unsigned int iRows,
		unsigned int iCols,
		FOVPreferSelected* pPreferSelectedFov)
	{
		// Create image
		Image* pImage;
		if(GetMosaicSet()->IsBayerPattern())
		{
			pImage = new ColorImage(BGR, false);
		}
		else
		{
			pImage = new Image();
		}

		double dRes = GetMosaicSet()->GetNominalPixelSizeX();
		ImgTransform inputTransform;
		inputTransform.Config(dRes, dRes, 0, dRes*iStartRowInCad, dRes*iStartColInCad);	
		pImage->Configure(iCols, iRows, iPixelSpan, inputTransform, inputTransform, false, pBuf);

		bool bFlag = GetImagePatch(
			pImage, 
			0,
			iCols-1,
			0,
			iRows-1,
			pPreferSelectedFov,
			iStartRowInCad,
			iStartColInCad);

		delete pImage;
		return(bFlag);
	}

	// Get a morphed image patch
	// pImage: inout, the image hold the patch, memeory is allocated before pass in
	// iLeft, iRight, iTop and iBottom: Roi in pixel
	// pPreferSelectedFov: inout, the prefer Fovs and selected Fovs 
	// pHeighImage: input, height image (it's origin should match pImage's origin)
	// dHeightResolution: gray level resolution of height image
	// PupilDistance: Pupil distance in meter
	bool MosaicLayer::GetImagePatch(
		Image* pImage, 
		unsigned int iLeft,
		unsigned int iRight,
		unsigned int iTop,
		unsigned int iBottom,
		FOVPreferSelected* pPreferSelectedFov,
		unsigned int iStartRowInCad,
		unsigned int iStartColInCad)
	{
		// valication check
		if(pImage == NULL) 
			return(false);

		if(iLeft>iRight || iTop>iBottom)
			return(false);

		int iChannels = pImage->GetBytesPerPixel();
		if(!GetMosaicSet()->IsBayerPattern())
		{
			if(iChannels != 1)
				return(false);
		}
		else
		{
			if(iChannels != 3)
				return(false);
		}

	// Calculate trig range and camera range for morph
		// ROI in world
		DRect worldRoi;
		pImage->ImageToWorld(iTop, iLeft, &worldRoi.xMin, &worldRoi.yMin);
		pImage->ImageToWorld(iBottom, iRight, &worldRoi.xMax, &worldRoi.yMax);

		// Fovs are organized into grid group to simplity the logic
		// However, the panel rotation need to be small for this approach
		// The grid boundarys are calculated if it is necessary
		if(!_bGridBoundaryValid)
			CalculateGridBoundary();

		int iStartInvTrig=-1, iEndInvTrig=-1;
		for(int iInvTrig = 0; iInvTrig < (int)_numTriggers; iInvTrig++)
		{ 
			if(_pdGridXBoundary[2*iInvTrig]<worldRoi.xMin && 
				worldRoi.xMin<_pdGridXBoundary[2*iInvTrig+1])
			{
				iStartInvTrig = iInvTrig;
				break;
			}
		}
		for(int iInvTrig = _numTriggers-1; iInvTrig >= 0; iInvTrig--)
		{
			if(_pdGridXBoundary[2*iInvTrig]<worldRoi.xMax && 
				worldRoi.xMax<_pdGridXBoundary[2*iInvTrig+1])
			{
				iEndInvTrig = iInvTrig;
				break;
			}
		}
		
		int iStartCam=-1, iEndCam=-1;
		for(int iCam = 0; iCam < (int)_numCameras; iCam++)
		{
			if(_pdGridYBoundary[2*iCam]<worldRoi.yMin && 
				worldRoi.yMin<_pdGridYBoundary[2*iCam+1])
			{
				iStartCam = iCam;
				break;
			}
		}
		for(int iCam = _numCameras-1; iCam >= 0; iCam--)
		{
			if(_pdGridYBoundary[2*iCam]<worldRoi.yMax && 
				worldRoi.yMax<_pdGridYBoundary[2*iCam+1])
			{
				iEndCam = iCam;
				break;
			}
		}
		
	// Calcuatle ROI of output image for each FOV that may be used 
		int iInvTrigCount = iEndInvTrig - iStartInvTrig + 1;
		int iCamCount = iEndCam - iStartCam + 1;

		int* piPixelRowBoundary = new int[iInvTrigCount+1]; // Contain start Rows
		int* piPixelColBoundary = new int[iCamCount+1];		// contain start Cols

		// If more than 2 triggers are covered
		if(iInvTrigCount > 2)
		{
			// No FOV preferance is used
			pPreferSelectedFov->selectedTB = NOPREFERTB;
			
			piPixelRowBoundary[0]= iTop;
			for(int iInvTrig = iStartInvTrig+1; iInvTrig<=iEndInvTrig; iInvTrig++)
			{
				double dStartRow, dEndRow, dTemp;
				if(iInvTrig == iStartInvTrig+1)
				{
					pImage->WorldToImage(_pdGridXBoundary[2*iInvTrig], 0, &dStartRow, &dTemp);
					int iStartRow = (int)dStartRow+1; // +1 to compensate the clip error 
					if(iStartRow < (int)iTop) iStartRow = iTop;
					if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
					piPixelRowBoundary[1]= iStartRow;
				}
				pImage->WorldToImage(_pdGridXBoundary[2*iInvTrig+1], 0, &dEndRow, &dTemp);
				int iStartRow = (int)dEndRow+1;	// +1 to next startRow
				if(iStartRow < (int)iTop) iStartRow = iTop;
				if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
				piPixelRowBoundary[(iInvTrig-iStartInvTrig)+1]= iStartRow;
			}
		}

		// if 2 triggers are convered, need consider Fov preferance
		if(iInvTrigCount == 2)
		{
			piPixelRowBoundary[0] = iTop;
			piPixelRowBoundary[2] = iBottom+1;
			// if no Fov preferance
			if(pPreferSelectedFov->preferTB == NOPREFERTB)
			{	
				int iStartRow;
				if(fabs(_pdGridXBoundary[iStartInvTrig*2]-worldRoi.xMin) <
					fabs(_pdGridXBoundary[iEndInvTrig*2+1]-worldRoi.xMax))
				{
					// Roi near start inverse trigger
					pPreferSelectedFov->selectedTB = TOPFOV;
					double dEndRow, dTemp;
					pImage->WorldToImage(_pdGridXBoundary[2*iStartInvTrig+1], 0, &dEndRow, &dTemp);
					iStartRow = (int)dEndRow + 1;
				}
				else
				{	// roi near end inverse trigger
					pPreferSelectedFov->selectedTB = BOTTOMFOV;
					double dStartRow, dTemp;
					pImage->WorldToImage(_pdGridXBoundary[2*iEndInvTrig], 0, &dStartRow, &dTemp);
					iStartRow = (int)dStartRow+1;
				}
				if(iStartRow < (int)iTop) iStartRow = iTop;
				if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
				piPixelRowBoundary[1]= iStartRow;
			}
			// Prefer top Fov
			if(pPreferSelectedFov->preferTB == TOPFOV)
			{
				if(_pdGridXBoundary[iStartInvTrig*2] < worldRoi.xMin &&
					worldRoi.xMax < _pdGridXBoundary[iStartInvTrig*2+1])
				{
					// Roi is totally inside x grid of prefered Fov  
					pPreferSelectedFov->selectedTB = pPreferSelectedFov->preferTB;
					piPixelRowBoundary[1] = iBottom+1;
				}
				else
				{
					// Roi is not totally inside x grid of prefered Fov
					if(_pdGridXBoundary[iEndInvTrig*2] < worldRoi.xMin &&
						worldRoi.xMax < _pdGridXBoundary[iEndInvTrig*2+1])
					{	
						// However Roi is totally inside x grid of none prefered Fov
						// the no prefered one will be selected
						pPreferSelectedFov->selectedTB = BOTTOMFOV;
						piPixelRowBoundary[1] = iTop;
					}
					else
					{	// Roi is not totally inside both x grid
						// Choice the prefered one as much as possible
						pPreferSelectedFov->selectedTB = pPreferSelectedFov->preferTB;
						double dEndRow, dTemp;
						pImage->WorldToImage(_pdGridXBoundary[2*iStartInvTrig+1], 0, &dEndRow, &dTemp);
						int iStartRow = (int)dEndRow + 1;
						if(iStartRow < (int)iTop) iStartRow = iTop;
						if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
						piPixelRowBoundary[1]= iStartRow;
					}
				}
			}
			// Prefer bottom Fov 
			if(pPreferSelectedFov->preferTB == BOTTOMFOV)
			{
				if(_pdGridXBoundary[iEndInvTrig*2] < worldRoi.xMin &&
					worldRoi.xMax < _pdGridXBoundary[iEndInvTrig*2+1])
				{
					// Roi is totally inside x grid of prefered Fov  
					pPreferSelectedFov->selectedTB = pPreferSelectedFov->preferTB;
					piPixelRowBoundary[1] = iTop;
				}
				else
				{
					// Roi is not totally inside x grid of prefered Fov
					if(_pdGridXBoundary[iStartInvTrig*2] < worldRoi.xMin &&
						worldRoi.xMax < _pdGridXBoundary[iStartInvTrig*2+1])
					{	
						// However Roi is totally inside x grid of none prefered Fov
						// the no prefered one will be selected
						pPreferSelectedFov->selectedTB = TOPFOV;
						piPixelRowBoundary[1] = iBottom+1;
					}
					else
					{	// Roi is not totally inside both x grid
						// Choice the prefered one as much as possible
						pPreferSelectedFov->selectedTB = pPreferSelectedFov->preferTB;
						double dStartRow, dTemp;
						pImage->WorldToImage(_pdGridXBoundary[2*iEndInvTrig], 0, &dStartRow, &dTemp);
						int iStartRow = (int)dStartRow + 1;
						if(iStartRow < (int)iTop) iStartRow = iTop;
						if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
						piPixelRowBoundary[1]= iStartRow;
					}
				}
			}
		}
		// If only covered by one trigger
		if(iInvTrigCount == 1)
		{
			pPreferSelectedFov->selectedTB = NOPREFERTB;
			piPixelRowBoundary[0] = iTop;
			piPixelRowBoundary[1] = iBottom+1;
		}

//***********************************************************************
		// If more than 2 cameras are covered
		if(iCamCount > 2)
		{
			// No FOV preferance is used
			pPreferSelectedFov->selectedLR = NOPREFERLR;
			
			piPixelColBoundary[0]= iLeft;
			for(int iCam = iStartCam+1; iCam<=iEndCam; iCam++)
			{
				double dStartCol, dEndCol, dTemp;
				if(iCam == iStartCam+1)
				{
					pImage->WorldToImage(0, _pdGridYBoundary[2*iCam], &dTemp, &dStartCol);
					int iStartCol = (int)dStartCol+1; // +1 to compensate the clip error 
					if(iStartCol < (int)iLeft) iStartCol = iLeft;
					if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
					piPixelColBoundary[1]= iStartCol;
				}
				pImage->WorldToImage(0, _pdGridYBoundary[2*iCam+1], &dTemp, &dEndCol);
				int iStartCol = (int)dEndCol+1;	// +1 to next startCol
				if(iStartCol < (int)iLeft) iStartCol = iLeft;
				if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
				piPixelColBoundary[(iCam-iStartCam)+1]= iStartCol;
			}
		}

		// if 2 cameras are convered, need consider Fov preferance
		if(iCamCount == 2)
		{
			piPixelColBoundary[0] = iLeft;
			piPixelColBoundary[2] = iRight+1;
			// if no Fov preferance
			if(pPreferSelectedFov->preferLR == NOPREFERLR)
			{	
				int iStartCol;
				if(fabs(_pdGridYBoundary[iStartCam*2]-worldRoi.yMin) <
					fabs(_pdGridYBoundary[iEndCam*2+1]-worldRoi.yMax))
				{
					// Roi near start camera
					pPreferSelectedFov->selectedLR = LEFTFOV;
					double dEndCol, dTemp;
					pImage->WorldToImage(0, _pdGridYBoundary[2*iStartCam+1], &dTemp, &dEndCol);
					iStartCol = (int)dEndCol + 1;
				}
				else
				{	// roi near end camera
					pPreferSelectedFov->selectedLR = RIGHTFOV;
					double dStartCol, dTemp;
					pImage->WorldToImage(0, _pdGridYBoundary[2*iEndCam], &dTemp, &dStartCol);
					iStartCol = (int)dStartCol+1;
				}
				if(iStartCol < (int)iLeft) iStartCol = iLeft;
				if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
				piPixelColBoundary[1]= iStartCol;
			}
			// Prefer left Fov
			if(pPreferSelectedFov->preferLR == LEFTFOV)
			{
				if(_pdGridYBoundary[iStartCam*2] < worldRoi.yMin &&
					worldRoi.yMax < _pdGridYBoundary[iStartCam*2+1])
				{
					// Roi is totally inside y grid of prefered Fov  
					pPreferSelectedFov->selectedLR = pPreferSelectedFov->preferLR;
					piPixelColBoundary[1] = iRight+1;
				}
				else
				{
					// Roi is not totally inside y grid of prefered Fov
					if(_pdGridYBoundary[iEndCam*2] < worldRoi.yMin &&
						worldRoi.yMax < _pdGridYBoundary[iEndCam*2+1])
					{	
						// However Roi is totally inside y grid of none prefered Fov
						// the no prefered one will be selected
						pPreferSelectedFov->selectedLR = RIGHTFOV;
						piPixelColBoundary[1] = iLeft;
					}
					else
					{	// Roi is not totally inside both y grid
						// Choice the prefered one as much as possible
						pPreferSelectedFov->selectedLR = pPreferSelectedFov->preferLR;
						double dEndCol, dTemp;
						pImage->WorldToImage(0, _pdGridYBoundary[2*iStartCam+1], &dTemp, &dEndCol);
						int iStartCol = (int)dEndCol + 1;
						if(iStartCol < (int)iLeft) iStartCol = iLeft;
						if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
						piPixelColBoundary[1]= iStartCol;
					}
				}
			}
			// Prefer right Fov 
			if(pPreferSelectedFov->preferLR == RIGHTFOV)
			{
				if(_pdGridYBoundary[iEndCam*2] < worldRoi.yMin &&
					worldRoi.yMax < _pdGridYBoundary[iEndCam*2+1])
				{
					// Roi is totally inside y grid of prefered Fov  
					pPreferSelectedFov->selectedLR = pPreferSelectedFov->preferLR;
					piPixelColBoundary[1] = iLeft;
				}
				else
				{
					// Roi is not totally inside y grid of prefered Fov
					if(_pdGridYBoundary[iStartCam*2] < worldRoi.yMin &&
						worldRoi.yMax < _pdGridYBoundary[iStartCam*2+1])
					{	
						// However Roi is totally inside x grid of none prefered Fov
						// the no prefered one will be selected
						pPreferSelectedFov->selectedLR = LEFTFOV;
						piPixelColBoundary[1] = iRight+1;
					}
					else
					{	// Roi is not totally inside both y grid
						// Choice the prefered one as much as possible
						pPreferSelectedFov->selectedLR = pPreferSelectedFov->preferLR;
						double dStartCol, dTemp;
						pImage->WorldToImage(0, _pdGridYBoundary[2*iEndCam], &dTemp, &dStartCol);
						int iStartCol = (int)dStartCol + 1;
						if(iStartCol < (int)iLeft) iStartCol = iLeft;
						if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
						piPixelColBoundary[1]= iStartCol;
					}
				}
			}
		}
		// If only covered by one camera
		if(iCamCount == 1)
		{
			pPreferSelectedFov->selectedLR = NOPREFERLR;
			piPixelColBoundary[0] = iLeft;
			piPixelColBoundary[1] = iRight+1;
		}

		// Create height image
		Image* pHeightImage = NULL;
		double dHeightResolution=0, dPupilDistance=0;
		if(_pHeightInfo != 0)
		{
			double dRes = GetMosaicSet()->GetNominalPixelSizeX();
			ImgTransform inputTransform;
			inputTransform.Config(dRes, dRes, 0, dRes*iStartRowInCad, dRes*iStartColInCad);

			pHeightImage = new Image();
			pHeightImage->Configure(
				iRight-iLeft+1,
				iBottom-iTop+1,
				_pHeightInfo->iHeightSpan,
				inputTransform,
				inputTransform,
				false,
				_pHeightInfo->pHeightBuf + _pHeightInfo->iHeightSpan*iStartRowInCad + iStartColInCad);

			dHeightResolution = _pHeightInfo->dHeightResolution;
			dPupilDistance = _pHeightInfo->dPupilDistance;
		}

		// Morph to create image patch
		for(int iInvTrig= iStartInvTrig; iInvTrig<= iEndInvTrig; iInvTrig++)
		{
			for(int iCam = iStartCam; iCam<=iEndCam; iCam++)
			{
				UIRect roi;
				roi.FirstRow = piPixelRowBoundary[iInvTrig-iStartInvTrig];
				roi.LastRow = piPixelRowBoundary[iInvTrig-iStartInvTrig+1]-1;
				roi.FirstColumn = piPixelColBoundary[iCam-iStartCam];
				roi.LastColumn = piPixelColBoundary[iCam-iStartCam+1]-1;
				if(roi.LastRow < roi.FirstRow  || roi.LastColumn < roi.FirstColumn)
					continue;

				Image* pFovImg = GetImage(iCam, _numTriggers-1-iInvTrig);
				pImage->MorphFrom(pFovImg, roi, pHeightImage, dHeightResolution, dPupilDistance);
			}
		}

		delete [] piPixelRowBoundary;
		delete [] piPixelColBoundary;

		if(pHeightImage !=NULL)
			delete pHeightImage; 

		return(true);
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

		return pTile->GetImagPtr();
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
			if(_pTileArray[i].GetImagPtr() != NULL)
				_pTileArray[i].GetImagPtr()->SetTransform(_pTileArray[i].GetImagPtr()->GetNominalTransform());
			_maskImages[i].SetTransform(_maskImages[i].GetNominalTransform());
		}	
		_bIsMaskImgValid = false;
		_stitchedImageValid = false;
		_bGridBoundaryValid = false;
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
				pdCenY[iCam] += GetTile(iCam, iTrig)->GetImagPtr()->CenterY();
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
				pdCenX[iTrig] += GetTile(iCam, iTrig)->GetImagPtr()->CenterX();
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
			_maskImages[i].SetTransform(_pTileArray[i].GetImagPtr()->GetTransform());
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

	// Set information for component height
	void MosaicLayer::SetComponentHeightInfo(				
		unsigned char* pHeightBuf,		// Component height image buf
		unsigned int iHeightSpan,		// Component height image span
		double dHeightResolution,		// Height resolution in grey level (meter/grey level)
		double dPupilDistance)			// SIM pupil distance (meter))
	{
		if(_pHeightInfo == NULL)
			_pHeightInfo = new ComponentHeightInfo();

		_pHeightInfo->pHeightBuf = pHeightBuf;
		_pHeightInfo->iHeightSpan = iHeightSpan;
		_pHeightInfo->dHeightResolution = dHeightResolution;
		_pHeightInfo->dPupilDistance = dPupilDistance;
	}

	// for debug
	Image* MosaicLayer::GetGreyStitchedImage(bool bRecreate)
	{
		Image* pTempImg = GetStitchedImage(bRecreate);

		if(pTempImg->GetBytesPerPixel()==1)
			return(pTempImg);
		else
		{
			_pGreyStitchedImage = new Image();		

			((ColorImage*)pTempImg)->Color2Luminance(_pGreyStitchedImage);

			return(_pGreyStitchedImage);
		}
	}
}