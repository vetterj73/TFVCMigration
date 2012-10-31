#include "StdAfx.h"
#include "MosaicLayer.h"
#include "MosaicSet.h"
#include "MosaicTile.h"
#include "MorphJob.h"
#include "JobManager.h"
#include "ColorImage.h"
#include <math.h>

using namespace CyberJob;
namespace MosaicDM 
{

#pragma region  Constructor

	MosaicLayer::MosaicLayer()
	{
		_pMosaicSet = NULL;
		_pTileArray = NULL;
		_pStitchedImage = NULL;
		_numCameras = 0;
		_numTriggers = 0;
		_bAlignWithCAD = false;
		_bAlignWithFiducial = true;
		_stitchedImageValid = false;

		_piStitchGridRows = NULL;
		_piStitchGridCols = NULL;

		_bGridBoundaryValid = false;
		_pdGridXBoundary = NULL;
		_pdGridYBoundary = NULL;

		_pHeightInfo = NULL;
		
		// For debug 
		_pGreyStitchedImage = NULL;
		_bXShift = false;
	}

	MosaicLayer::~MosaicLayer(void)
	{
		if(_pTileArray != NULL) 
			delete[] _pTileArray;

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

		_piStitchGridRows = new int[_numTriggers+1];
		_piStitchGridCols = new int[_numCameras+1]; 

		_pdGridXBoundary = new double[_numTriggers*2];
		_pdGridYBoundary = new double[_numCameras*2];

		for(unsigned int i=0; i<numTiles; i++)
		{
			// @todo - set X/Y center offsets nominally...
			_pTileArray[i].Initialize(this, 0.0, 0.0);
		}
	}

#pragma endregion 

#pragma  region Create stitched image 

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

	void MosaicLayer::SetStitchedBuffer(unsigned char *pBuffer)
	{
		AllocateStitchedImageIfNecessary();
		int iSize = _pMosaicSet->GetObjectWidthInPixels()*_pMosaicSet->GetObjectLengthInPixels();
		if (GetMosaicSet()->IsBayerPattern() && !GetMosaicSet()->IsSkipDemosaic())
			iSize *= 3;
		memcpy(_pStitchedImage->GetBuffer(), pBuffer, iSize);
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
		double dxShift = 0;
		if(_bXShift)
			dxShift = +100*_pMosaicSet->GetNominalPixelSizeX();
		inputTransform.Config(_pMosaicSet->GetNominalPixelSizeX(), 
			_pMosaicSet->GetNominalPixelSizeY(), 0, dxShift, 0);
			
		unsigned int iNumRows = _pMosaicSet->GetObjectWidthInPixels();
		unsigned int iNumCols = _pMosaicSet->GetObjectLengthInPixels();

		_pStitchedImage->Configure(iNumCols, iNumRows, iNumCols, inputTransform, inputTransform, true);
		_pStitchedImage->CalInverseTransform();
	}

	// Camera centers in Y of world space
	void MosaicLayer::CameraCentersInY(double* pdCenY)
	{
		// Panel area in X
		double dPanelXMin = 0;
		double dPanelXMax = _pMosaicSet->GetObjectWidthInMeters();	

		// Min overlap needed for FOV to be considered in calculation
		double dMinOverlapInX = _pMosaicSet->GetImageHeightInPixels() * _pMosaicSet->GetNominalPixelSizeX() * 0.4;

		for(unsigned int iCam=0; iCam<GetNumberOfCameras(); iCam++)
		{
			pdCenY[iCam] = 0;
			int iCount = 0;
			for(unsigned int iTrig=0; iTrig<GetNumberOfTriggers(); iTrig++)
			{
				// if more than one trigger, ignore FOV has a little overlap with panel in X
				if(GetNumberOfTriggers() > 1)
				{	
					DRect rect = GetImage(iTrig, iCam)->GetBoundBoxInWorld();
					double dOverlapXMin = rect.xMin > dPanelXMin ? rect.xMin : dPanelXMin;
					double dOverlapXMax = rect.xMax < dPanelXMax ? rect.xMax : dPanelXMax;
					if(dOverlapXMax - dOverlapXMin < dMinOverlapInX)
						continue;
				}

				pdCenY[iCam] += GetTile(iTrig, iCam)->GetImagPtr()->CenterY();
				iCount++;
			}
			pdCenY[iCam] /= iCount;
		}
	}

	// Trigger centesr in  X of world space 
	void MosaicLayer::TriggerCentersInX(double* pdCenX)
	{
		// Panel area in Y
		double dPanelYMin = 0;
		double dPanelYMax = _pMosaicSet->GetObjectLengthInMeters();	

		// Min overlap needed for FOV to be considered in calculation
		double dMinOverlapInY = _pMosaicSet->GetImageWidthInPixels() * _pMosaicSet->GetNominalPixelSizeY() * 0.4;
		
		for(unsigned int iTrig=0; iTrig<GetNumberOfTriggers(); iTrig++)
		{
			pdCenX[iTrig] = 0;
			int iCount = 0;
			for(unsigned int iCam=0; iCam<GetNumberOfCameras(); iCam++)
			{	
				// if more than one camera, ignore FOV has a little overlap with panel in Y
				if(GetNumberOfCameras() > 1)
				{	
					DRect rect = GetImage(iTrig, iCam)->GetBoundBoxInWorld();
					double dOverlapYMin = rect.yMin > dPanelYMin ? rect.yMin : dPanelYMin;
					double dOverlapYMax = rect.yMax < dPanelYMax ? rect.yMax : dPanelYMax;
					if(dOverlapYMax - dOverlapYMin < dMinOverlapInY)
						continue;
				}

				pdCenX[iTrig] += GetTile(iTrig, iCam)->GetImagPtr()->CenterX();
				iCount++;
			}
			pdCenX[iTrig] /= iCount++;
		}
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
		CyberJob::JobManager jm(buf, _pMosaicSet->GetThreadNumber());
		vector<MorphJob*> morphJobs;
		// Morph each Fov to create stitched panel image
		for(unsigned int iTrig=0; iTrig<iNumTrigs; iTrig++)
		{
			for(unsigned int iCam=0; iCam<iNumCams; iCam++)
			{
				Image* pFOV = GetImage(iTrig, iCam);

				// For test purpose, it will modify the image buffer, 
				// It isn't optimized
				// When bayer patern without demosaic
				//if(_pMosaicSet->IsBayerPattern() && _pMosaicSet->IsSkipDemosaic())
				//	pFOV->Bayer2Lum((BayerType)_pMosaicSet->GetBayerType());

				bool bDemosaic = _pMosaicSet->IsBayerPattern() && _pMosaicSet->IsSkipDemosaic();
				MorphJob *pJob = new MorphJob(_pStitchedImage, pFOV,
					(unsigned int)_piStitchGridCols[iCam], (unsigned int)_piStitchGridRows[iTrig+1], 
					(unsigned int)(_piStitchGridCols[iCam+1]-1), (unsigned int)(_piStitchGridRows[iTrig]-1),
					bDemosaic, (BayerType)_pMosaicSet->GetBayerType(),	_pMosaicSet->IsGaussianDemosaic(),
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

#pragma endregion

#pragma region image patch 
	// Calculate all grid boundarys
	// Fovs are organize into grid group 
	bool MosaicLayer::CalculateAllGridBoundary()
	{
		if(_bGridBoundaryValid) return(true);

		bool bFlag = CalculateGridBoundary(
			0, _numTriggers-1,
			0, _numCameras-1,
			_pdGridXBoundary, _pdGridYBoundary);

		if(bFlag)
			_bGridBoundaryValid = true;

		return(bFlag);
	}

	// Calculate grid boundarys
	// Fovs are organize into grid group 
	bool MosaicLayer::CalculateGridBoundary(
		int iBeginInvTrig, int iEndInvTrig,
		int iBeginCam, int iEndCam,
		double* pdGridXBoundary, double* pdGridYBoundary)
	{		
		// Panel Rect
		DRect panelRect;
		panelRect.xMin = 0;
		panelRect.yMin = 0;
		panelRect.xMax = _pMosaicSet->GetObjectWidthInMeters();
		panelRect.yMax = _pMosaicSet->GetObjectLengthInMeters();	

		// Min overlap needed for FOV to be considered in calculation
		double dMinOverlapInX = _pMosaicSet->GetImageHeightInPixels() * _pMosaicSet->GetNominalPixelSizeX() * 0.4;
		double dMinOverlapInY = _pMosaicSet->GetImageWidthInPixels() * _pMosaicSet->GetNominalPixelSizeY() * 0.4;

		// Boundary for each inverse trigger (x in world)
		for(int iInvTrig=iBeginInvTrig; iInvTrig<=iEndInvTrig; iInvTrig++)
		{
			double dTopBound, dBottomBound;
			bool bFirstCam = true;
			for(int iCam=iBeginCam; iCam<=iEndCam; iCam++)
			{
				DRect rect = GetImage(_numTriggers-1-iInvTrig, iCam)->GetBoundBoxInWorld();
				
				// if more than one camera, ignore FOV has a little overlap with panel in Y
				if(iBeginCam != iEndCam)
				{
					double dOverlapYMin = rect.yMin > panelRect.yMin ? rect.yMin : panelRect.yMin;
					double dOverlapYMax = rect.yMax < panelRect.yMax ? rect.yMax : panelRect.yMax;
					if(dOverlapYMax - dOverlapYMin < dMinOverlapInY)
						continue;
				}

				if(bFirstCam)
				{
					bFirstCam = false;
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
			int iIndex = iInvTrig - iBeginInvTrig;
			pdGridXBoundary[iIndex*2] = dTopBound;
			pdGridXBoundary[iIndex*2+1] = dBottomBound;
		}

		// Boundary for each camera (y in world)
		for(int iCam=iBeginCam; iCam<=iEndCam; iCam++)
		{
			double dLeftBound, dRightBound;
			bool bFirstTrig = true;
			for(int iTrig=iBeginInvTrig; iTrig<=iEndInvTrig; iTrig++)
			{
				DRect rect = GetImage(_numTriggers-1-iTrig, iCam)->GetBoundBoxInWorld();

				// if more than one trigger, ignore FOV has a little overlap with panel in X
				if(iBeginInvTrig != iEndInvTrig)
				{
					double dOverlapXMin = rect.xMin > panelRect.xMin ? rect.xMin : panelRect.xMin;
					double dOverlapXMax = rect.xMax < panelRect.xMax ? rect.xMax : panelRect.xMax;
					if(dOverlapXMax - dOverlapXMin < dMinOverlapInX)
						continue;
				}

				if(bFirstTrig)
				{
					bFirstTrig = false;
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
			int iIndex = iCam - iBeginCam;
			pdGridYBoundary[iIndex*2] = dLeftBound;
			pdGridYBoundary[iIndex*2+1] = dRightBound;
		}

		return(true);
	}

	// Calculate the FOV used for image patch
	// worldRoi: Image patch ROI in world
	// bAjustForHeight: whether adjust for height
	// piStartInvTrig and piEndInvTrig: output, range of triggers in inverse index
	// piStartCam and piEndCam: output, range of cameras
	// pdPatchGridXBoundary and pdPatchGridYBoundary: output, grid boundary for image patch
	void MosaicLayer::CalculateFOVsForImagePatch(
		Image* pImage,		
		int iLeft, int iRight,
		int iTop, int iBottom,
		DRect worldRoi, bool bAjustForHeight,
		int* piStartInvTrig, int* piEndInvTrig,
		int* piStartCam, int* piEndCam,
		double* pdPatchGridXBoundary, double* pdPatchGridYBoundary)
	{
		// Fovs are organized into grid group to simplify the logic
		// However, the panel rotation need to be small for this approach
		// The grid boundarys are calculated if it is necessary
		if(!_bGridBoundaryValid)
			CalculateAllGridBoundary();

		// Max FOV boundary shift brought by height adjustment
		double dMaxHeightShiftX = 0;
		double dMaxHeightShiftY = 0; 
		if(bAjustForHeight && _pHeightInfo != 0)
		{
			double dRes = GetMosaicSet()->GetNominalPixelSizeX();
			int iStartRowInCad = (int)(pImage->GetTransform().GetItem(2)/dRes + 0.5) + iLeft;
			int iStartColInCad = (int)(pImage->GetTransform().GetItem(5)/dRes + 0.5) + iTop;	
			int iRows = iBottom - iTop + 1;
			int iCols = iRight - iLeft + 1;

			int iMax = 0;
			unsigned char* pLine = _pHeightInfo->pHeightBuf + _pHeightInfo->iHeightSpan*iStartRowInCad + iStartColInCad;
			for(int iy = 0; iy < iRows; iy++)
			{
				for(int ix = 0; ix < iCols; ix++)
				{
					if (iMax <(int)pLine[ix])
						iMax = (int)pLine[ix];
				}
				pLine += _pHeightInfo->iHeightSpan;
			}

			double dMaxHeight = iMax * _pHeightInfo->dHeightResolution;
			double dScale = 1.2;
			double dFovInX = 3.3e-2;
			double dFovInY = 4.4e-2;
			dMaxHeightShiftX = dMaxHeight/_pHeightInfo->dPupilDistance * dFovInX/2 * dScale;
			dMaxHeightShiftY = dMaxHeight/_pHeightInfo->dPupilDistance* dFovInY/2 * dScale;
		}

		// Grid boundary for whole panel, adjusted by height
		for(unsigned int i=0; i<_numTriggers; i++)
		{
			pdPatchGridXBoundary[2*i] = _pdGridXBoundary[2*i] + dMaxHeightShiftX; 
			pdPatchGridXBoundary[2*i+1] = _pdGridXBoundary[2*i+1] - dMaxHeightShiftX; 
			// Make sure adjacent FOV boundary has overlap
			if(i>0)
			{
				if(pdPatchGridXBoundary[2*i-1] < pdPatchGridXBoundary[2*i])
				{
					double dHalf = (pdPatchGridXBoundary[2*i-1] + pdPatchGridXBoundary[2*i])/2;
					pdPatchGridXBoundary[2*i-1] = dHalf;
					pdPatchGridXBoundary[2*i] = dHalf;
				}
			}
		}

		for(unsigned int i=0; i<_numCameras; i++)
		{
			pdPatchGridYBoundary[2*i] = _pdGridYBoundary[2*i] + dMaxHeightShiftY; 
			pdPatchGridYBoundary[2*i+1] = _pdGridYBoundary[2*i+1] - dMaxHeightShiftY; 
			// Make sure adjacent FOV boundary has overlap
			if(i>0)
			{
				if(pdPatchGridYBoundary[2*i-1] < pdPatchGridYBoundary[2*i])
				{
					double dHalf = (pdPatchGridYBoundary[2*i-1] + pdPatchGridYBoundary[2*i])/2;
					pdPatchGridYBoundary[2*i-1] = dHalf;
					pdPatchGridYBoundary[2*i] = dHalf;
				}
			}
		}

		// Calculate start and end inverse triggers
		int iStartInvTrig=-1, iEndInvTrig=-1;
		if(worldRoi.xMin <= pdPatchGridXBoundary[0])
		{
			iStartInvTrig = 0;
		}
		else if(worldRoi.xMin >= pdPatchGridXBoundary[2*_numTriggers-1])
		{
			iStartInvTrig = _numTriggers-1;
		}
		else
		{
			for(int iInvTrig = 0; iInvTrig < (int)_numTriggers; iInvTrig++)
			{ 
				if(pdPatchGridXBoundary[2*iInvTrig]<worldRoi.xMin && 
					worldRoi.xMin<=pdPatchGridXBoundary[2*iInvTrig+1])
				{
					iStartInvTrig = iInvTrig;
					break;
				}
			}
		}
		
		if(worldRoi.xMax <= pdPatchGridXBoundary[0])
		{
			iEndInvTrig = 0;
		}
		else if(worldRoi.xMax >= pdPatchGridXBoundary[2*_numTriggers-1])
		{
			iEndInvTrig = _numTriggers-1;
		}
		else
		{
			for(int iInvTrig = _numTriggers-1; iInvTrig >= 0; iInvTrig--)
			{
				if(pdPatchGridXBoundary[2*iInvTrig]<worldRoi.xMax && 
					worldRoi.xMax<=pdPatchGridXBoundary[2*iInvTrig+1])
				{
					iEndInvTrig = iInvTrig;
					break;
				}
			}
		}
		
		// Calculate start and end cameras
		int iStartCam=-1, iEndCam=-1;
		if(worldRoi.yMin <= pdPatchGridYBoundary[0])
		{
			iStartCam = 0;
		}
		else if (worldRoi.yMin >= pdPatchGridYBoundary[2*_numCameras-1])
		{
			iStartCam = _numCameras-1;
		}
		else
		{
			for(int iCam = 0; iCam < (int)_numCameras; iCam++)
			{
				if(pdPatchGridYBoundary[2*iCam]<worldRoi.yMin && 
					worldRoi.yMin<=pdPatchGridYBoundary[2*iCam+1])
				{
					iStartCam = iCam;
					break;
				}
			}
		}

		if(worldRoi.yMax <= pdPatchGridYBoundary[0])
		{
			iEndCam = 0;
		}
		else if (worldRoi.yMax >= pdPatchGridYBoundary[2*_numCameras-1])
		{
			iEndCam = _numCameras-1;
		}
		else
		{
			for(int iCam = _numCameras-1; iCam >= 0; iCam--)
			{
				if(pdPatchGridYBoundary[2*iCam]<worldRoi.yMax && 
					worldRoi.yMax<=pdPatchGridYBoundary[2*iCam+1])
				{
					iEndCam = iCam;
					break;
				}
			}
		}

		// Grid boundary for selected triggers and cameras only
		CalculateGridBoundary(
			iStartInvTrig, iEndInvTrig,
			iStartCam, iEndCam,
			pdPatchGridXBoundary + iStartInvTrig*2, 
			pdPatchGridYBoundary + iStartCam*2);

		// Adjust grid boundary by height
		if(dMaxHeightShiftX>0)
		{
			for(int i=iStartInvTrig; i<=iEndInvTrig; i++)
			{
				pdPatchGridXBoundary[2*i] += dMaxHeightShiftX; 
				pdPatchGridXBoundary[2*i+1] -= dMaxHeightShiftX; 
				// Make sure adjacent FOV boundary has overlap
				if(i>iStartInvTrig)
				{
					if(pdPatchGridXBoundary[2*i-1] < pdPatchGridXBoundary[2*i])
					{
						// Should never reach here
						double dHalf = (pdPatchGridXBoundary[2*i-1] + pdPatchGridXBoundary[2*i])/2;
						pdPatchGridXBoundary[2*i-1] = dHalf;
						pdPatchGridXBoundary[2*i] = dHalf;
					}
				}
			}
		}

		for(int i=iStartCam; i<=iEndCam; i++)
		{
			pdPatchGridYBoundary[2*i] += dMaxHeightShiftY; 
			pdPatchGridYBoundary[2*i+1] -= dMaxHeightShiftY; 
			// Make sure adjacent FOV boundary has overlap
			if(i>iStartCam)
			{
				if(pdPatchGridYBoundary[2*i-1] < pdPatchGridYBoundary[2*i])
				{
					// Should never reach here
					double dHalf = (pdPatchGridYBoundary[2*i-1] + pdPatchGridYBoundary[2*i])/2;
					pdPatchGridYBoundary[2*i-1] = dHalf;
					pdPatchGridYBoundary[2*i] = dHalf;
				}
			}
		}

		// Output value
		*piStartInvTrig = iStartInvTrig; 
		*piEndInvTrig = iEndInvTrig;
		*piStartCam = iStartCam;
		*piEndCam = iEndCam;
	}

	// Calcualte boundary in Row for FOV used for image patch
	// pImage and word ROI: input, patch image and its ROi in world space
	// iTop and iBottom: patch image and its location in CAD
	// pdPatchGridXBoundary: grid boundary for image patch
	// iStartInvTrig and iEndInvTrig: trigger range
	// pPreferSelectedFov: inout, the prefer Fovs and selected Fovs 
	// piPixelRowBoundary: ouput, boundary in Row for FOV used for image patch
	void MosaicLayer::CalPatchFOVRowBoundary(
		Image* pImage, 
		DRect worldRoi,
		unsigned int iTop,
		unsigned int iBottom,
		const double* pdPatchGridXBoundary,
		int iStartInvTrig,
		int iEndInvTrig,
		FOVPreferSelected* pPreferSelectedFov,
		int* piPixelRowBoundary)
	{
		int iInvTrigCount = iEndInvTrig - iStartInvTrig + 1;

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
					pImage->WorldToImage(pdPatchGridXBoundary[2*iInvTrig], 0, &dStartRow, &dTemp);
					int iStartRow = (int)dStartRow+1; // +1 to compensate the clip error 
					if(iStartRow < (int)iTop) iStartRow = iTop;
					if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
					piPixelRowBoundary[1]= iStartRow;
				}
				pImage->WorldToImage(pdPatchGridXBoundary[2*iInvTrig+1], 0, &dEndRow, &dTemp);
				int iStartRow = (int)dEndRow+1;	// +1 to next startRow
				if(iStartRow < (int)iTop) iStartRow = iTop;
				if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
				piPixelRowBoundary[(iInvTrig-iStartInvTrig)+1]= iStartRow;
			}
		}
		// if 2 triggers are convered, need consider Fov preferance
		else if(iInvTrigCount == 2)
		{
			piPixelRowBoundary[0] = iTop;
			piPixelRowBoundary[2] = iBottom+1;
			// if no Fov preferance
			if(pPreferSelectedFov->preferTB == NOPREFERTB)
			{	
				int iStartRow;
				if(fabs(pdPatchGridXBoundary[iStartInvTrig*2]-worldRoi.xMin) <
					fabs(pdPatchGridXBoundary[iEndInvTrig*2+1]-worldRoi.xMax))
				{
					// Roi near start inverse trigger
					pPreferSelectedFov->selectedTB = TOPFOV;
					double dEndRow, dTemp;
					pImage->WorldToImage(pdPatchGridXBoundary[2*iStartInvTrig+1], 0, &dEndRow, &dTemp);
					iStartRow = (int)dEndRow + 1;
				}
				else
				{	// roi near end inverse trigger
					pPreferSelectedFov->selectedTB = BOTTOMFOV;
					double dStartRow, dTemp;
					pImage->WorldToImage(pdPatchGridXBoundary[2*iEndInvTrig], 0, &dStartRow, &dTemp);
					iStartRow = (int)dStartRow+1;
				}
				if(iStartRow < (int)iTop) iStartRow = iTop;
				if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
				piPixelRowBoundary[1]= iStartRow;
			}
			// Prefer top Fov
			else if(pPreferSelectedFov->preferTB == TOPFOV)
			{
				if(pdPatchGridXBoundary[iStartInvTrig*2] < worldRoi.xMin &&
					worldRoi.xMax < pdPatchGridXBoundary[iStartInvTrig*2+1])
				{
					// Roi is totally inside x grid of prefered Fov  
					pPreferSelectedFov->selectedTB = pPreferSelectedFov->preferTB;
					piPixelRowBoundary[1] = iBottom+1;
				}
				else
				{
					// Roi is not totally inside x grid of prefered Fov
					if(pdPatchGridXBoundary[iEndInvTrig*2] < worldRoi.xMin &&
						worldRoi.xMax < pdPatchGridXBoundary[iEndInvTrig*2+1])
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
						pImage->WorldToImage(pdPatchGridXBoundary[2*iStartInvTrig+1], 0, &dEndRow, &dTemp);
						int iStartRow = (int)dEndRow + 1;
						if(iStartRow < (int)iTop) iStartRow = iTop;
						if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
						piPixelRowBoundary[1]= iStartRow;
					}
				}
			}
			// Prefer bottom Fov 
			else if(pPreferSelectedFov->preferTB == BOTTOMFOV)
			{
				if(pdPatchGridXBoundary[iEndInvTrig*2] < worldRoi.xMin &&
					worldRoi.xMax < pdPatchGridXBoundary[iEndInvTrig*2+1])
				{
					// Roi is totally inside x grid of prefered Fov  
					pPreferSelectedFov->selectedTB = pPreferSelectedFov->preferTB;
					piPixelRowBoundary[1] = iTop;
				}
				else
				{
					// Roi is not totally inside x grid of prefered Fov
					if(pdPatchGridXBoundary[iStartInvTrig*2] < worldRoi.xMin &&
						worldRoi.xMax < pdPatchGridXBoundary[iStartInvTrig*2+1])
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
						pImage->WorldToImage(pdPatchGridXBoundary[2*iEndInvTrig], 0, &dStartRow, &dTemp);
						int iStartRow = (int)dStartRow + 1;
						if(iStartRow < (int)iTop) iStartRow = iTop;
						if(iStartRow > (int)iBottom+1) iStartRow = iBottom+1;
						piPixelRowBoundary[1]= iStartRow;
					}
				}
			}
		}
		// If only covered by one trigger
		else if(iInvTrigCount == 1)
		{
			pPreferSelectedFov->selectedTB = NOPREFERTB;
			piPixelRowBoundary[0] = iTop;
			piPixelRowBoundary[1] = iBottom+1;
		}
	}

	
	// Calcualte boundary in Column for FOV used for image patch
	// pImage and word ROI: input, patch image and its ROi in world space
	// iLeft and iRight: patch image and its location in CAD
	// pdPatchGridYBoundary: grid boundary for image patch
	// iStartCam and iEndCam: cameras range
	// pPreferSelectedFov: inout, the prefer Fovs and selected Fovs 
	// piPixelColBoundary: ouput, boundary in column for FOV used for image patch
	void MosaicLayer::CalPatchFOVColumnBoundary(
		Image* pImage, 
		DRect worldRoi,
		unsigned int iLeft,
		unsigned int iRight,
		const double* pdPatchGridYBoundary,
		int iStartCam,
		int iEndCam,
		FOVPreferSelected* pPreferSelectedFov,
		int* piPixelColBoundary)
	{
		int iCamCount = iEndCam - iStartCam + 1;

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
					pImage->WorldToImage(0, pdPatchGridYBoundary[2*iCam], &dTemp, &dStartCol);
					int iStartCol = (int)dStartCol+1; // +1 to compensate the clip error 
					if(iStartCol < (int)iLeft) iStartCol = iLeft;
					if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
					piPixelColBoundary[1]= iStartCol;
				}
				pImage->WorldToImage(0, pdPatchGridYBoundary[2*iCam+1], &dTemp, &dEndCol);
				int iStartCol = (int)dEndCol+1;	// +1 to next startCol
				if(iStartCol < (int)iLeft) iStartCol = iLeft;
				if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
				piPixelColBoundary[(iCam-iStartCam)+1]= iStartCol;
			}
		}
		// if 2 cameras are convered, need consider Fov preferance
		else if(iCamCount == 2)
		{
			piPixelColBoundary[0] = iLeft;
			piPixelColBoundary[2] = iRight+1;
			// if no Fov preferance
			if(pPreferSelectedFov->preferLR == NOPREFERLR)
			{	
				int iStartCol;
				if(fabs(pdPatchGridYBoundary[iStartCam*2]-worldRoi.yMin) <
					fabs(pdPatchGridYBoundary[iEndCam*2+1]-worldRoi.yMax))
				{
					// Roi near start camera
					pPreferSelectedFov->selectedLR = LEFTFOV;
					double dEndCol, dTemp;
					pImage->WorldToImage(0, pdPatchGridYBoundary[2*iStartCam+1], &dTemp, &dEndCol);
					iStartCol = (int)dEndCol + 1;
				}
				else
				{	// roi near end camera
					pPreferSelectedFov->selectedLR = RIGHTFOV;
					double dStartCol, dTemp;
					pImage->WorldToImage(0, pdPatchGridYBoundary[2*iEndCam], &dTemp, &dStartCol);
					iStartCol = (int)dStartCol+1;
				}
				if(iStartCol < (int)iLeft) iStartCol = iLeft;
				if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
				piPixelColBoundary[1]= iStartCol;
			}
			// Prefer left Fov
			else if(pPreferSelectedFov->preferLR == LEFTFOV)
			{
				if(pdPatchGridYBoundary[iStartCam*2] < worldRoi.yMin &&
					worldRoi.yMax < pdPatchGridYBoundary[iStartCam*2+1])
				{
					// Roi is totally inside y grid of prefered Fov  
					pPreferSelectedFov->selectedLR = pPreferSelectedFov->preferLR;
					piPixelColBoundary[1] = iRight+1;
				}
				else
				{
					// Roi is not totally inside y grid of prefered Fov
					if(pdPatchGridYBoundary[iEndCam*2] < worldRoi.yMin &&
						worldRoi.yMax < pdPatchGridYBoundary[iEndCam*2+1])
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
						pImage->WorldToImage(0, pdPatchGridYBoundary[2*iStartCam+1], &dTemp, &dEndCol);
						int iStartCol = (int)dEndCol + 1;
						if(iStartCol < (int)iLeft) iStartCol = iLeft;
						if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
						piPixelColBoundary[1]= iStartCol;
					}
				}
			}
			// Prefer right Fov 
			else if(pPreferSelectedFov->preferLR == RIGHTFOV)
			{
				if(pdPatchGridYBoundary[iEndCam*2] < worldRoi.yMin &&
					worldRoi.yMax < pdPatchGridYBoundary[iEndCam*2+1])
				{
					// Roi is totally inside y grid of prefered Fov  
					pPreferSelectedFov->selectedLR = pPreferSelectedFov->preferLR;
					piPixelColBoundary[1] = iLeft;
				}
				else
				{
					// Roi is not totally inside y grid of prefered Fov
					if(pdPatchGridYBoundary[iStartCam*2] < worldRoi.yMin &&
						worldRoi.yMax < pdPatchGridYBoundary[iStartCam*2+1])
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
						pImage->WorldToImage(0, pdPatchGridYBoundary[2*iEndCam], &dTemp, &dStartCol);
						int iStartCol = (int)dStartCol + 1;
						if(iStartCol < (int)iLeft) iStartCol = iLeft;
						if(iStartCol > (int)iRight+1) iStartCol = iRight+1;
						piPixelColBoundary[1]= iStartCol;
					}
				}
			}
		}
		// If only covered by one camera
		else if(iCamCount == 1)
		{
			pPreferSelectedFov->selectedLR = NOPREFERLR;
			piPixelColBoundary[0] = iLeft;
			piPixelColBoundary[1] = iRight+1;
		}
	}

	// Get a morphed image patch
	// pBuf: inout, the buffer hold the patch, memeory is allocated before pass in
	// iPixelSpan: buffer's span in pixel
	// iStartCol, iWidth, iStartRow and iHeight: Roi in pixel of CAD space
	// pPreferSelectedFov: inout, the prefer Fovs and selected Fovs 
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
		if(GetMosaicSet()->IsBayerPattern() && !GetMosaicSet()->IsSkipDemosaic())
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
			pPreferSelectedFov);

		delete pImage;
		return(bFlag);
	}

	// Get a morphed image patch
	// pImage: inout, the image hold the patch, memeory is allocated before pass in
	// iLeft, iRight, iTop and iBottom: Roi in pixel of pImage
	// pPreferSelectedFov: inout, the prefer Fovs and selected Fovs 
	bool MosaicLayer::GetImagePatch(
		Image* pImage, 
		unsigned int iLeft,
		unsigned int iRight,
		unsigned int iTop,
		unsigned int iBottom,
		FOVPreferSelected* pPreferSelectedFov)
	{
		// valication check
		if(pImage == NULL) 
			return(false);

		if(iLeft>iRight || iTop>iBottom)
			return(false);

		int iChannels = pImage->GetBytesPerPixel();
		if(!GetMosaicSet()->IsBayerPattern() && !GetMosaicSet()->IsSkipDemosaic())
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

		double *pdPatchGridXBoundary = new double[2*_numTriggers];
		double *pdPatchGridYBoundary = new double[2*_numCameras];

		int iStartInvTrig, iEndInvTrig, iStartCam, iEndCam;
		CalculateFOVsForImagePatch(
			pImage,
			iLeft, iRight,
			iTop, iBottom,
			worldRoi, true,
			&iStartInvTrig, &iEndInvTrig,
			&iStartCam, &iEndCam,
			pdPatchGridXBoundary, pdPatchGridYBoundary);
		
	// Calcuatle ROI of output image for each FOV that may be used 
		int iInvTrigCount = iEndInvTrig - iStartInvTrig + 1;
		int iCamCount = iEndCam - iStartCam + 1;

		int* piPixelRowBoundary = new int[iInvTrigCount+1]; // Contain start Rows
		int* piPixelColBoundary = new int[iCamCount+1];		// contain start Cols

		// For Row boundarys of FOVs
		CalPatchFOVRowBoundary(
			pImage, worldRoi,
			iTop, iBottom,
			pdPatchGridXBoundary,
			iStartInvTrig,	iEndInvTrig,
			pPreferSelectedFov,
			piPixelRowBoundary);

		// For Column boundarys of FOVs
		CalPatchFOVColumnBoundary(
			pImage, worldRoi,
			iLeft, iRight,
			pdPatchGridYBoundary,
			iStartCam, iEndCam,
			pPreferSelectedFov,
			piPixelColBoundary);

		// Create height image
		Image* pHeightImage = NULL;
		double dHeightResolution=0, dPupilDistance=0;
		if(pPreferSelectedFov->bAdjustForHeight && _pHeightInfo != 0)
		{
			double dRes = GetMosaicSet()->GetNominalPixelSizeX();
			int iStartRowInCad = (int)(pImage->GetTransform().GetItem(2)/dRes +0.5);
			int iStartColInCad = (int)(pImage->GetTransform().GetItem(5)/dRes +0.5);	
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

				Image* pFovImg = GetImage(_numTriggers-1-iInvTrig, iCam);
				pImage->MorphFrom(pFovImg, true, roi, pHeightImage, dHeightResolution, dPupilDistance);
			}
		}

		delete [] piPixelRowBoundary;
		delete [] piPixelColBoundary;

		delete [] pdPatchGridXBoundary;
		delete [] pdPatchGridYBoundary;

		if(pHeightImage !=NULL)
			delete pHeightImage; 

		return(true);
	}

#pragma endregion

#pragma region FOV  and mask related

	MosaicTile* MosaicLayer::GetTile(unsigned int triggerIndex, unsigned int cameraIndex)
	{
		if(cameraIndex<0 || cameraIndex>=GetNumberOfCameras() || triggerIndex<0 || triggerIndex>=GetNumberOfTriggers())
			return NULL;

		return &_pTileArray[cameraIndex*GetNumberOfTriggers()+triggerIndex];
	}

	Image* MosaicLayer::GetImage(int triggerIndex, int cameraIndex)
	{
		MosaicTile* pTile = GetTile(triggerIndex, cameraIndex);
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
		}	
		_stitchedImageValid = false;
		_bGridBoundaryValid = false;
	}

	bool MosaicLayer::AddRawImage(unsigned char *pBuffer, unsigned int cameraIndex, unsigned int triggerIndex)
	{
		MosaicTile* pTile = GetTile(triggerIndex, cameraIndex);
		if(pTile == NULL)
			return false;

		return pTile->SetRawImageBuffer(pBuffer);
	}

	bool MosaicLayer::AddYCrCbImage(unsigned char *pBuffer, unsigned int cameraIndex, unsigned int triggerIndex)
	{
		MosaicTile* pTile = GetTile(triggerIndex, cameraIndex);
		if(pTile == NULL)
			return false;

		return pTile->SetYCrCbImageBuffer(pBuffer);
	}

#pragma endregion


#pragma region debug

	// for debug
	Image* MosaicLayer::GetGreyStitchedImage(bool bRecreate)
	{
		Image* pTempImg = GetStitchedImage(bRecreate);

		if(pTempImg->GetBytesPerPixel()==1)
			return(pTempImg);
		else
		{
			if(_pGreyStitchedImage != NULL)
			{
				delete _pGreyStitchedImage;
				_pGreyStitchedImage = NULL;
			}
			_pGreyStitchedImage = new Image();		

			((ColorImage*)pTempImg)->Color2Luminance(_pGreyStitchedImage);

			return(_pGreyStitchedImage);
		}
	}

	void MosaicLayer::SetXShift(bool bValue) 
	{
		_bXShift = bValue;
		_stitchedImageValid = false; // next stitch image need recalcualte
	};

#pragma endregion
}