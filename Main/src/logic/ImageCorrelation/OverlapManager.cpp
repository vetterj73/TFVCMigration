#include "OverlapManager.h"
#include "Logger.h"
#include "RenderShape.h"
#include "CorrelationParameters.h"
#include "MosaicLayer.h"
#include "EquationWeights.h"
#include <direct.h> //_mkdir

#pragma region constructor and reset
OverlapManager::OverlapManager(
	MosaicSet* pMosaicSet,
	Panel* pPanel,
	unsigned int numThreads)
{	
	_pJobManager = new CyberJob::JobManager("Overlap", numThreads);
	_pMosaicSet = pMosaicSet;
	_pPanel = pPanel;

	/*
	int k=2;
	double dS = 1e-3;
	double dAngle = 0;
	CrossFeature cross(k++, dS*100, dS*100, dAngle, dS*1, dS*1, dS*0.3, dS*0.3);
	_pPanel->AddFiducial(&cross);
	
	DiamondFeature diamond(k++, dS*100, dS*110, dAngle, dS*1, dS*2);
	_pPanel->AddFiducial(&diamond);
	
	DiamondFrameFeature diamondFrame(k++, dS*100, dS*120, dAngle, dS*1, dS*2, dS*0.2);
	_pPanel->AddFiducial(&diamondFrame);
	
	DiscFeature disc(k++, dS*100, dS*130, dS*1);
	_pPanel->AddFiducial(&disc);

	DonutFeature donut(k++, dS*100, dS*140, dS*0.7, dS*1);
	_pPanel->AddFiducial(&donut);
	
	RectangularFeature rect(k++, dS*110, dS*100, dAngle, dS*1, dS*2);
	_pPanel->AddFiducial(&rect);
	
	RectangularFrameFeature rectFrame(k++, dS*110, dS*110, dAngle, dS*1, dS*2, dS*0.2);
	_pPanel->AddFiducial(&rectFrame);
	
	TriangleFeature triangle(k++, dS*110, dS*120, dAngle, dS*1, dS*1, dS*0.5);
	_pPanel->AddFiducial(&triangle);
	
	EquilateralTriangleFrameFeature eqTriangleFrame(k++, dS*110, dS*130, dAngle, dS*1, dS*0.2);
	_pPanel->AddFiducial(&eqTriangleFrame);
	
	CheckerPatternFeature check1(k++, dS*110, dS*140, dAngle-90, dS*1, dS*2);
	_pPanel->AddFiducial(&check1);
	
	CheckerPatternFeature check2(k++, dS*110, dS*140, dAngle, dS*1, dS*2);
	_pPanel->AddFiducial(&check2);
	//*/

	// Valid panel area in world space 
	_validRect.xMin = 0;
	_validRect.yMin = 0;
	_validRect.xMax = pPanel->xLength();
	_validRect.yMax = pPanel->yLength();

	// Create Cad and Mask images if their buffers are provided
	unsigned int iNumRows = pPanel->GetNumPixelsInX();
	unsigned int iNumCols = pPanel->GetNumPixelsInY();
	bool bCreateOwnBuf = false;
	unsigned int iBytePerPixel = 1;
		// create image transform
	double t[3][3];
	t[0][0] = pPanel->GetPixelSizeX();
	t[0][1] = 0;
	t[0][2] = _validRect.xMin;
	t[1][0] = 0;
	t[1][1] = pPanel->GetPixelSizeX();
	t[1][2] = _validRect.yMin;
	t[2][0] = 0;
	t[2][1] = 0;
	t[2][2] = 1;
	ImgTransform trans(t);
	
	// Create Cad image if it is necessary
	_pCadImg = NULL;
	if(IsCadImageNeeded())
	{
		if(_pPanel->GetCadBuffer() != NULL)
		{
			_pCadImg = new Image(iNumCols, iNumRows, iNumCols, iBytePerPixel, 
				trans, trans, bCreateOwnBuf, _pPanel->GetCadBuffer());
			//_pCadImg->Save("C:\\Temp\\cad.bmp");
		}
	}

	// Create Panel Mask image
	_pPanelMaskImg = NULL;
	if(IsMaskImageNeeded())
	{
		unsigned char* pMaskBuf = _pPanel->GetMaskBuffer(CorrelationParametersInst.iMaskExpansionFromCad);
		if(pMaskBuf != NULL)
		{
			_pPanelMaskImg = new Image(iNumCols, iNumRows, iNumCols, iBytePerPixel, 
				trans, trans, bCreateOwnBuf, pMaskBuf);
			//_pPanelMaskImg->Save("C:\\Temp\\mask.bmp");
		}
	}

	// Control parameter
	_iMinOverlapSize = CorrelationParametersInst.iMinOverlapSize;

	// Calculate max number of cameras and triggers for mossiac images
	unsigned int i, j;
	_iNumCameras=0;
	_iNumTriggers=0;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		if (_iNumCameras < _pMosaicSet->GetLayer(i)->GetNumberOfCameras())
			_iNumCameras = _pMosaicSet->GetLayer(i)->GetNumberOfCameras();

		if (_iNumTriggers < _pMosaicSet->GetLayer(i)->GetNumberOfTriggers())
			_iNumTriggers = _pMosaicSet->GetLayer(i)->GetNumberOfTriggers();
	}

	// Create 3D arrays for storage of overlaps
	_fovFovOverlapLists = new list<FovFovOverlap>**[_pMosaicSet->GetNumMosaicLayers()];
	_cadFovOverlapLists = new list<CadFovOverlap>**[_pMosaicSet->GetNumMosaicLayers()];
	_fidFovOverlapLists = new list<FidFovOverlap>**[_pMosaicSet->GetNumMosaicLayers()];
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		_fovFovOverlapLists[i] = new list<FovFovOverlap>*[_iNumTriggers];
		_cadFovOverlapLists[i] = new list<CadFovOverlap>*[_iNumTriggers];
		_fidFovOverlapLists[i] = new list<FidFovOverlap>*[_iNumTriggers];

		for(j=0; j<_iNumTriggers; j++)
		{
			_fovFovOverlapLists[i][j] = new list<FovFovOverlap>[_iNumCameras];
			_cadFovOverlapLists[i][j] = new list<CadFovOverlap>[_iNumCameras];
			_fidFovOverlapLists[i][j] = new list<FidFovOverlap>[_iNumCameras];
		}
	}

	// Initial fiducial result set
	_pFidResultsSet = NULL;
	_pFidResultsSet = new PanelFiducialResultsSet(_pPanel->NumberOfFiducials());

	// Create FovFov overlaps
	CreateFovFovOverlaps();

	// Create CadFov overlaps
	if(_pCadImg != NULL) CreateCadFovOverlaps();
	
	// Create Fiducial Fov overlaps
	if(CorrelationParametersInst.fidSearchMethod == FIDVSFINDER)	
		VsFinderCorrelation::Instance().Config(_pPanel->GetPixelSizeX(), iNumCols, iNumRows);

	_pFidImages = NULL;
	CreateFidFovOverlaps();

	// Decide the stage to calculate mask
	CalMaskCreationStage();
}

OverlapManager::~OverlapManager(void)
{
	delete _pJobManager;

	// Release 3D arrays for storage
	unsigned int i, j;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		for(j=0; j<_iNumTriggers; j++)
		{
			delete [] _fovFovOverlapLists[i][j];
			delete [] _cadFovOverlapLists[i][j];
			delete [] _fidFovOverlapLists[i][j];
		}

		delete [] _fovFovOverlapLists[i];
		delete [] _cadFovOverlapLists[i];
		delete [] _fidFovOverlapLists[i];
	}
	delete [] _fovFovOverlapLists;
	delete [] _cadFovOverlapLists;
	delete [] _fidFovOverlapLists;

	if(_pFidResultsSet != NULL)
		delete _pFidResultsSet;

	if(_pFidImages != NULL)
		delete [] _pFidImages;
}

// Reset mosaic images and overlaps for new panel inspection 
bool OverlapManager::ResetforNewPanel()
{
	// Reset all mosaic images for new panel inspection
	unsigned int i, iCam , iTrig;

	// Reset all overlaps for new panel inspection
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		for(iTrig=0; iTrig<_iNumTriggers; iTrig++)
		{
			for(iCam=0; iCam<_iNumCameras; iCam++)
			{
				// Reset Fov and Fov overlap
				list<FovFovOverlap>* pFovFovList = &_fovFovOverlapLists[i][iTrig][iCam];
				for(list<FovFovOverlap>::iterator j=pFovFovList->begin(); j!=pFovFovList->end(); j++)
				{
					j->Reset();
				}

				// Reset Cad and Fov overlap
				list<CadFovOverlap>* pCadFovList = &_cadFovOverlapLists[i][iTrig][iCam]; 
				for(list<CadFovOverlap>::iterator j=pCadFovList->begin(); j!=pCadFovList->end(); j++)
				{
					j->Reset();
				}

				// Reset Fid and Fov overlap
				list<FidFovOverlap>* pFidFovList = &_fidFovOverlapLists[i][iTrig][iCam];
				for(list<FidFovOverlap>::iterator j=pFidFovList->begin(); j!=pFidFovList->end(); j++)
				{
					j->Reset();
				}
			} //iCam
		} // iTrig
	} // i

	return(true);
}


// Whether cad image is needed
bool OverlapManager::IsCadImageNeeded()
{
	unsigned int i;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		MosaicLayer* pLayer = _pMosaicSet->GetLayer(i);
		if(pLayer->IsAlignWithCad())	// If use Cad
			return(true);
	}

	return(false); //If not use Cad
}

// Whether mask image is needed
bool OverlapManager::IsMaskImageNeeded()
{
	unsigned int i, j;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		for(j=i; j<_pMosaicSet->GetNumMosaicLayers(); j++)
		{
			CorrelationFlags *pFlags = _pMosaicSet->GetCorrelationFlags(i, j);
			if(pFlags->GetMaskNeeded())	// If use mask
				return(true);
		}
	}

	return(false);	// If not use mask
}

#pragma endregion

#pragma region FovFovOverlaps
// Create Fov overlap for two illuminations
bool OverlapManager::CreateFovFovOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2)
{
	MosaicLayer *pLayer1 = _pMosaicSet->GetLayer(iIndex1);
	MosaicLayer *pLayer2 = _pMosaicSet->GetLayer(iIndex2);

	// Correlation setting between two layers (illuminations)
	CorrelationFlags *pFlags = _pMosaicSet->GetCorrelationFlags(iIndex1, iIndex2);
	bool bCamCam = pFlags->GetCameraToCamera();
	bool bTrigTrig = pFlags->GetTriggerToTrigger();
	bool bMask = pFlags->GetMaskNeeded();
	if(bMask) 		// Prepare mask images
		pLayer1->PrepareMaskImages();
	bool bApplyCorSizeUpLimit = pFlags->GetApplyCorrelationAreaSizeUpLimit();
	
	// Camera centers in Y of world space and trigger centers in X of world space 
	// Attention: Trgger center X is dcreaseing with trigger index
	// Cam center Y is increasing with camera index
	unsigned int iNumTrigs1 = pLayer1->GetNumberOfTriggers();
	unsigned int iNumCams1 = pLayer1->GetNumberOfCameras();
	double* pdCenX1 = new double[iNumTrigs1];
	double* pdCenY1 = new double[iNumCams1];
	pLayer1->TriggerCentersInX(pdCenX1);
	pLayer1->CameraCentersInY(pdCenY1);
	double dSpanTrig = (pdCenX1[0] - pdCenX1[iNumTrigs1-1])/(iNumTrigs1-1);
	double dSpanCam = (pdCenY1[iNumCams1-1] - pdCenY1[0])/(iNumCams1-1);
	
	unsigned int iNumTrigs2 = pLayer2->GetNumberOfTriggers();
	unsigned int iNumCams2 = pLayer2->GetNumberOfCameras();
	double* pdCenX2 = new double[iNumTrigs2];
	double* pdCenY2 = new double[iNumCams2];
	pLayer2->TriggerCentersInX(pdCenX2);
	pLayer2->CameraCentersInY(pdCenY2);

	// For each image in first layer (illuminaiton)
	unsigned int iCam1, iTrig1;
	int iCam2, iTrig2; // Must be integer to avoid for cycle error
	for(iTrig1 = 0; iTrig1<iNumTrigs1; iTrig1++)
	{
		for(iCam1 = 0; iCam1<iNumCams1; iCam1++)
		{
			// Cam to cam (Col to col) overlap
			if(bCamCam)
			{			
				// The nearest trigger
				int iTrigIndex=-1;	
				double dMinDis = 1.0*dSpanTrig; // If nearest trigger is far away, just ignore
				for(iTrig2=0; iTrig2<(int)iNumTrigs2; iTrig2++)
				{	
					double dis = fabs(pdCenX1[iTrig1] -pdCenX2[iTrig2]);	
					if(dMinDis > dis)
					{
						dMinDis = dis;
						iTrigIndex = iTrig2;
					}		
				}
				
				if(iTrigIndex >= 0)
				{
					// The nearest left image in the nearest trigger with a distance bigger than 0.5 span
					int iLeftCamIndex=-1;
					dMinDis = 1.2*dSpanCam;
					// From left to right (cam++)
					for(iCam2 = 0; iCam2<(int)iNumCams2; iCam2++)
					{	// pdCenY1/2[i] increases with i	
						double dis = pdCenY1[iCam1] -pdCenY2[iCam2];
						// >= 0.5 span to avoid Fov of the same camera
						if(dis > 0.5*dSpanCam && dis <dMinDis)
						{
							iLeftCamIndex = iCam2;
							dMinDis = dis;
						}
					}
					// The nearest right image in the nearest trigger with a distance bigger than 0.5 span
					int iRightCamIndex=-1;
					dMinDis = 1.2*dSpanCam;
					// From right to left (cam--)
					for(iCam2 = iNumCams2-1; iCam2>=0; iCam2--)
					{	// pdCenY1/2[i] increases with i
						double dis = -(pdCenY1[iCam1]-pdCenY2[iCam2]);
						if(dis > 0.5*dSpanCam  && dis <dMinDis)
						{
							iRightCamIndex = iCam2;
							dMinDis = dis;
							break;
						}
					}

					// Create left overlap and add to list 
					// If FOV are not from the same mosaic image  (avoid the same overlap add two times for a FOV)
					if(iLeftCamIndex>=0 && iIndex1!= iIndex2)
					{
						FovFovOverlap overlap(
							pLayer1, pLayer2,
							pair<unsigned int, unsigned int>(iCam1, iTrig1),
							pair<unsigned int, unsigned int>(iLeftCamIndex, iTrigIndex),
							_validRect, bApplyCorSizeUpLimit, bMask);

						if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
						{
							_fovFovOverlapLists[iIndex1][iTrig1][iCam1].push_back(overlap);
							_fovFovOverlapLists[iIndex2][iTrigIndex][iLeftCamIndex].push_back(overlap);
						}
					}
					// Create right overlap and add to list
					if(iRightCamIndex >= 0)
					{
						FovFovOverlap overlap(
							pLayer1, pLayer2,
							pair<unsigned int, unsigned int>(iCam1, iTrig1),
							pair<unsigned int, unsigned int>(iRightCamIndex, iTrigIndex),
							_validRect, bApplyCorSizeUpLimit, bMask);

						if(overlap.IsValid() && overlap.Columns()>_iMinOverlapSize && overlap.Rows()>_iMinOverlapSize)
						{
							_fovFovOverlapLists[iIndex1][iTrig1][iCam1].push_back(overlap);
							_fovFovOverlapLists[iIndex2][iTrigIndex][iRightCamIndex].push_back(overlap);
						}
					}
				}
			} //if(bCamCam)

			// Trig to trig (Row to row) overlap
			if(bTrigTrig)
			{
				// Find the nearest camera
				int iCamIndex = -1;
				double dMinDis = 0.6*dSpanCam; // If nearest camere is far away, ignore
				for(iCam2=0; iCam2<(int)iNumCams2; iCam2++)
				{
					double dis = fabs(pdCenY1[iCam1] -pdCenY2[iCam2]);
					
					if(dMinDis>dis && dis < 0.5*dSpanCam)
					{
						dMinDis = dis;
						iCamIndex = iCam2; 
					}
				}
				
				if(iCamIndex >= 0)
				{
					// Any nearby trigger of the nearest camera 
					for(iTrig2=0; iTrig2<(int)iNumTrigs2; iTrig2++)
					{
						
						double dis = fabs(pdCenX1[iTrig1] -pdCenX2[iTrig2]);
						bool bValid = false;
						if(iIndex1 != iIndex2) // For different mosaic images
						{
							if(dis<0.8*dSpanTrig)
								bValid = true;
						}
						else // for the same mosaic image (avoid the same overlap add two times for a FOV)
						{
							if(pdCenX1[iTrig1]>pdCenX2[iTrig2] && dis<1.2*dSpanTrig && dis>0.8*dSpanTrig)
								bValid = true;
						}

						if(bValid)
						{
							FovFovOverlap overlap(
								pLayer1, pLayer2,
								pair<unsigned int, unsigned int>(iCam1, iTrig1),
								pair<unsigned int, unsigned int>(iCamIndex, iTrig2),
								_validRect, bApplyCorSizeUpLimit, bMask);

							if(overlap.IsValid() && overlap.Columns()>_iMinOverlapSize && overlap.Rows()>_iMinOverlapSize)
							{
								_fovFovOverlapLists[iIndex1][iTrig1][iCam1].push_back(overlap);
								_fovFovOverlapLists[iIndex2][iTrig2][iCamIndex].push_back(overlap);
							}
						}
					}
				}
			} // Trig to Trig
		}
	}

	delete [] pdCenX1;
	delete [] pdCenY1;
	delete [] pdCenX2;
	delete [] pdCenY2;

	return true;
}

// Create Fov and Fov overlaps
void OverlapManager::CreateFovFovOverlaps()
{
	unsigned int i, j;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		for(j=i; j<_pMosaicSet->GetNumMosaicLayers(); j++)
		{
			CreateFovFovOverlapsForTwoIllum(i, j);
		}
	}
}

// Get FovFovOverlap list for certain Fov
list<FovFovOverlap>* OverlapManager::GetFovFovListForFov(
	unsigned int iMosaicIndex, 
	unsigned int iTrigIndex,
	unsigned int iCamIndex) const
{
	return(&_fovFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]);
}

#pragma endregion

#pragma region CadFovOverlaps
// Create Cad and Fov overlaps
void OverlapManager::CreateCadFovOverlaps()
{
	unsigned int i, iCam, iTrig;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		MosaicLayer* pLayer = _pMosaicSet->GetLayer(i);
		if(!pLayer->IsAlignWithCad()) // If not use Cad
			continue;

		// If use Cad
		unsigned int iNumCameras = pLayer->GetNumberOfCameras();
		unsigned int iNumTriggers = pLayer->GetNumberOfTriggers();
		for(iTrig=0; iTrig<iNumTriggers; iTrig++)
		{
			for(iCam=0; iCam<iNumCameras; iCam++)
			{
				CadFovOverlap overlap(
					pLayer,
					pair<unsigned int, unsigned int>(iCam, iTrig),
					_pCadImg,
					_validRect);

				if(overlap.IsValid() && overlap.Columns()>_iMinOverlapSize && overlap.Rows()>_iMinOverlapSize)
					_cadFovOverlapLists[i][iTrig][iCam].push_back(overlap);
			}
		}
	}
}

// Get CadFovOverlap list for certain Fov
list<CadFovOverlap>* OverlapManager::GetCadFovListForFov(
	unsigned int iMosaicIndex, 
	unsigned int iTrigIndex,
	unsigned int iCamIndex) const
{
	return(&_cadFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]);
}
#pragma endregion

#pragma region FidFovOverlap
// Create Fiducial images
void OverlapManager::CreateFiducialImage(Image* pImage, Feature* pFeature)
{
	double dScale = 1.0;						// The expansion scale for regoff
	if(CorrelationParametersInst.fidSearchMethod == FIDVSFINDER)
		dScale = 1.4;
	else if(CorrelationParametersInst.fidSearchMethod == FIDCYBERNGC) 
		dScale = 1.4;	// The expansion scale for template base NGC and vsfinder
	
	RenderFiducial(
		pImage, 
		pFeature, 
		_pPanel->GetPixelSizeX(), 
		dScale);	
}

// fiducial is placed in the exact center of the resulting image
// this is important as SqRtCorrelation (used to build least squares table)
// measures the positional difference between the center of the fiducial 
// image and the center of the FOV image segement.  Any information
// about offset of the fiducial from the center of the FOV is lost
// img: output, fiducial image,
// fid: input, fiducial feature 
// resolution: the pixel size, in meters
// dScale: the fiducial area expansion scale
void OverlapManager::RenderFiducial(
	Image* pImg, 
	Feature* pFid, 
	double resolution, 
	double dScale)
{
	// Area in world space
	double dExpandX = CorrelationParametersInst.dFiducialSearchExpansionX;
	double dExpandY = CorrelationParametersInst.dFiducialSearchExpansionY;
	double dHalfWidth = pFid->GetBoundingBox().Width()/2;
	double dHalfHeight= pFid->GetBoundingBox().Height()/2;
	double xMinImg = pFid->GetBoundingBox().Min().x - dHalfWidth*(dScale-1)  - dExpandX; 
	double yMinImg = pFid->GetBoundingBox().Min().y - dHalfHeight*(dScale-1) - dExpandY; 
	double xMaxImg = pFid->GetBoundingBox().Max().x + dHalfWidth*(dScale-1)  + dExpandX; 
	double yMaxImg = pFid->GetBoundingBox().Max().y + dHalfHeight*(dScale-1) + dExpandY;
	
	// The size of image in pixels
	unsigned int nRowsImg
			(unsigned int((1.0/resolution)*(xMaxImg-xMinImg)));	
	unsigned int nColsImg
			(unsigned int((1.0/resolution)*(yMaxImg-yMinImg)));	
	
	// Define transform for new fiducial image
	double t[3][3];
	t[0][0] = resolution;
	t[0][1] = 0;
	t[0][2] = pFid->GetCadX() - resolution * (nRowsImg-1) / 2.;
	
	t[1][0] = 0;
	t[1][1] = resolution;
	t[1][2] = pFid->GetCadY() - resolution * (nColsImg-1) / 2.;
	
	t[2][0] = 0;
	t[2][1] = 0;
	t[2][2] = 1;	
	
	ImgTransform trans(t);

	// Create and clean buffer
	pImg->Configure(nColsImg, nRowsImg, nColsImg, trans, trans, true);
	pImg->ZeroBuffer();

	// Fiducial drawing/render
	unsigned int grayValue = 255;
	int antiAlias = 1;
	RenderFeature(pImg, pFid, grayValue, antiAlias);
}

// Create Cyber Ngc template
int OverlapManager::CreateNgcFidTemplate(
	Image* pImage, 
	Feature* pFeature,
	bool bFidBrighterThanBackground,
	bool bFiducialAllowNegativeMatch)
{
	// Fiducial image serach expansion in Pixels
	int iPixelExpRow = (int)(CorrelationParametersInst.dFiducialSearchExpansionX/_pPanel->GetPixelSizeX());
	int iPixelExpCol = (int)(CorrelationParametersInst.dFiducialSearchExpansionY/_pPanel->GetPixelSizeY());

	// Template roi
	UIRect tempRoi;
	tempRoi.FirstColumn = iPixelExpCol;
	tempRoi.LastColumn = pImage->Columns()-1 - iPixelExpCol;
	tempRoi.FirstRow = iPixelExpRow;
	tempRoi.LastRow = pImage->Rows()-1 - iPixelExpRow;


	// Create template
	int iId = CyberNgcFiducialCorrelation::Instance().CreateNgcTemplate(pFeature, bFidBrighterThanBackground, bFiducialAllowNegativeMatch, pImage, tempRoi);
	if(iId < 0) 
	{
		LOG.FireLogEntry(LogTypeError, "OverlapManager::CreateNgcFidTemplates():Failed to create CyberNgc template");
		return(-1);
	}

	return(iId);
}


// Create Fiducial and Fov overlaps
void OverlapManager::CreateFidFovOverlaps()
{
	// Validation check
	unsigned int iNum = _pPanel->NumberOfFiducials();
	if(iNum <=0) 
	{
		LOG.FireLogEntry(LogTypeError, "OverlapManager::CreateFiducialImages():No fiducial avaliable for alignment");
		return;
	}

	// Set feature to result set 
	int iIndex = 0;
	for(FeatureListIterator iFid = _pPanel->beginFiducials(); iFid != _pPanel->endFiducials(); iFid++)
	{
		_pFidResultsSet->GetPanelFiducialResultsPtr(iIndex)->SetFeaure(iFid->second);
		iIndex++;
	}

	_pFidImages = new Image[iNum*2];

	unsigned int iLayer, iCam, iTrig, iCount=-1;	
	for(FeatureListIterator iFid = _pPanel->beginFiducials(); iFid != _pPanel->endFiducials(); iFid++)
	{
		iCount++;
		// Create a fiducial image
		CreateFiducialImage(&_pFidImages[iCount], iFid->second);
		/* for Debug
		string s;
		char cTemp[100];
		sprintf_s(cTemp, 100, "C:\\Temp\\Fid_%d.bmp", iCount);
		s.append(cTemp);
		_pFidImages[iCount].Save(s);
		//*/

		for(iLayer=0; iLayer<_pMosaicSet->GetNumMosaicLayers(); iLayer++)
		{
			MosaicLayer *pLayer = _pMosaicSet->GetLayer(iLayer);
			if(!pLayer->IsAlignWithFiducial()) // If not use Fiducial
				continue;

			bool bFidBrighter = pLayer->IsFiducialBrighterThanBackground();
			bool bAllowNegMatch = pLayer->IsFiducialAllowNegativeMatch();

			int iVsFinderTemplateId, iCyberNgcTemplateId;
			if(CorrelationParametersInst.fidSearchMethod == FIDVSFINDER)
			{
				iVsFinderTemplateId = VsFinderCorrelation::Instance().CreateVsTemplate(iFid->second, bFidBrighter, bAllowNegMatch);
				if(iVsFinderTemplateId < 0) 
				{
					LOG.FireLogEntry(LogTypeError, "OverlapManager::CreateFidFovOverlaps():Failed to create vsfinder template");
					return;
				}
			}
			else if(CorrelationParametersInst.fidSearchMethod == FIDCYBERNGC)
			{
				iCyberNgcTemplateId = CreateNgcFidTemplate(&_pFidImages[iCount], iFid->second, bFidBrighter, bAllowNegMatch);
				if(iCyberNgcTemplateId < 0) 
				{
					LOG.FireLogEntry(LogTypeError, "OverlapManager::CreateFidFovOverlaps():Failed to create cyberNgc template");
					return;
				}
			}

			unsigned int iNumCameras = pLayer->GetNumberOfCameras();
			unsigned int iNumTriggers = pLayer->GetNumberOfTriggers();
			for(iTrig=0; iTrig<iNumTriggers; iTrig++)
			{
				for(iCam=0; iCam<iNumCameras; iCam++)
				{
					FidFovOverlap overlap(
						pLayer,
						pair<unsigned int, unsigned int>(iCam, iTrig),
						&_pFidImages[iCount],
						_pFidImages[iCount].CenterX(),
						_pFidImages[iCount].CenterY(),
						_validRect);

					// Overlap validation check
					if(!overlap.IsValid())
						continue;

					// Make sure fiducial template area is inside overlap 
					int iSafePixels = 15;
					UIRect rectFid= overlap.GetCoarsePair()->GetSecondRoi();
					unsigned int iExpPixelsCol = (unsigned int)(CorrelationParametersInst.dFiducialSearchExpansionY/_pPanel->GetPixelSizeY());
					unsigned int iExpPixelsRow = (unsigned int)(CorrelationParametersInst.dFiducialSearchExpansionX/_pPanel->GetPixelSizeX());
					if(rectFid.FirstColumn >  iExpPixelsCol-iSafePixels ||
						rectFid.LastColumn < _pFidImages[iCount].Columns()-1 - (iExpPixelsCol-iSafePixels) || 
						rectFid.FirstRow >  iExpPixelsRow-iSafePixels ||
						rectFid.LastRow < _pFidImages[iCount].Rows()-1 - (iExpPixelsRow-iSafePixels))
						continue;

					if(CorrelationParametersInst.fidSearchMethod == FIDVSFINDER)
					{
						overlap.SetVsFinder(iVsFinderTemplateId);
					}
					else if(CorrelationParametersInst.fidSearchMethod == FIDCYBERNGC)
					{
						overlap.SetNgcFid(iCyberNgcTemplateId);
					}

					// Add overlap
					_fidFovOverlapLists[iLayer][iTrig][iCam].push_back(overlap);
					
					// Add to fiducial result set
					FidFovOverlap* pOvelap1 = &(*_fidFovOverlapLists[iLayer][iTrig][iCam].rbegin());
					_pFidResultsSet->GetPanelFiducialResultsPtr(iFid->second->GetId())->AddFidFovOvelapPoint(pOvelap1);
				}
			}
		}
	}
}

// Get FidFovOverlap list for certain Fov
list<FidFovOverlap>* OverlapManager::GetFidFovListForFov(
	unsigned int iMosaicIndex, 
	unsigned int iTrigIndex,
	unsigned int iCamIndex) const
{	
	return(&_fidFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]);
}

#pragma endregion

#pragma region Do Alignment

// Do alignment for a certain Fov
bool OverlapManager::DoAlignmentForFov(
	unsigned int iMosaicIndex, 
	unsigned int iTrigIndex,
	unsigned int iCamIndex)
{
	if(CorrelationParametersInst.bSaveOverlaps || CorrelationParametersInst.bSaveFiducialOverlaps)
	{
		_mkdir(CorrelationParametersInst.GetOverlapPath().c_str());
	}

	// process valid FovFov Overlap  
	list<FovFovOverlap>* pFovFovList = &_fovFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex];
	for(list<FovFovOverlap>::iterator i=pFovFovList->begin(); i!=pFovFovList->end(); i++)
	{
		if(i->IsReadyToProcess())
			_pJobManager->AddAJob((CyberJob::Job*)&*i);
	}

	// Process valid Cad Fov overalp
	list<CadFovOverlap>* pCadFovList = &_cadFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]; 
	for(list<CadFovOverlap>::iterator i=pCadFovList->begin(); i!=pCadFovList->end(); i++)
	{
		if(i->IsReadyToProcess())
			_pJobManager->AddAJob((CyberJob::Job*)&*i);
	}

	// Process valid fiducial Fov overlap
	list<FidFovOverlap>* pFidFovList = &_fidFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]; 
	for(list<FidFovOverlap>::iterator i=pFidFovList->begin(); i!=pFidFovList->end(); i++)
	{
		if(i->IsReadyToProcess())
		{ 
			// Let vsfinder run on a single thread (temporary solution)
			// PanelAligner::ImageAddedToMosaicCallback() works on single thread as Alan Claimed
			// so all the vsfinder will work on single thread as well
			// The solver will be filled after all the overlap (include fiducials) are calculated
			//i->Run(); 

			_pJobManager->AddAJob((CyberJob::Job*)&*i);
		}
	}

	return(true);
}

#pragma endregion

#pragma region Solver setup
// Decide after what layer is completed, mask images need to be created
void OverlapManager::CalMaskCreationStage()
{
	unsigned int i, j;
	unsigned iMin = _pMosaicSet->GetNumMosaicLayers()+10;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		for(j=i; j<_pMosaicSet->GetNumMosaicLayers(); j++)
		{
			if(_pMosaicSet->GetCorrelationFlags(i, j)->GetMaskNeeded())
			{
				if(i>j)
				{
					if(iMin>i) iMin = i;
				}
				else
				{
					if(iMin>j) iMin = j;
				}
			}
		}
	}

	if(iMin > _pMosaicSet->GetNumMosaicLayers())
		_iMaskCreationStage = -1;
	else
		_iMaskCreationStage = iMin;
}

unsigned int OverlapManager::MaxCorrelations() const
{
	unsigned int iFovFovCount = 0;
	unsigned int iCadFovCount = 0;
	unsigned int iFidFovCount = 0;

	unsigned int i, iTrig, iCam;
	for(i=0; i<_pMosaicSet->GetNumMosaicLayers(); i++)
	{
		for(iTrig=0; iTrig<_iNumTriggers; iTrig++)
		{
			for(iCam=0; iCam<_iNumCameras; iCam++)
			{
				// FovFov
				iFovFovCount += (unsigned int)_fovFovOverlapLists[i][iTrig][iCam].size();

				// CadFov
				iCadFovCount += (unsigned int)_cadFovOverlapLists[i][iTrig][iCam].size();

				//FidFov
				iFidFovCount += (unsigned int)_fidFovOverlapLists[i][iTrig][iCam].size();
			}
		}
	}

	iFovFovCount /=2;

	// Double check 3*3
	unsigned int iSum = CorrelationParametersInst.iFineMaxBlocksInRow * CorrelationParametersInst.iFineMaxBlocksInCol * iFovFovCount+iCadFovCount+iFidFovCount;
	return(iSum);
}

//Report possible Maximum correlation will be used by solver to create mask
unsigned int OverlapManager::MaxMaskCorrelations() const
{
	if(_iMaskCreationStage<=0)
		return(0);

	unsigned int* piIllumIndices = new unsigned int[_iMaskCreationStage];
	for(int i=0; i<_iMaskCreationStage; i++)
		piIllumIndices[i] = i;

	unsigned int iNum =MaxCorrelations(piIllumIndices, _iMaskCreationStage);

	delete [] piIllumIndices;

	return(iNum);
}

//Report possible maximum corrleaiton will be used by solver to create transforms
unsigned int OverlapManager::MaxCorrelations(unsigned int* piIllumIndices, unsigned int iNumIllums) const
{
	unsigned int iFovFovCount = 0;
	unsigned int iCadFovCount = 0;
	unsigned int iFidFovCount = 0;

	unsigned int i, iTrig, iCam;
	for(i=0; i<iNumIllums; i++)
	{
		for(iTrig=0; iTrig<_iNumTriggers; iTrig++)
		{
			for(iCam=0; iCam<_iNumCameras; iCam++)
			{
				// FovFov
				list<FovFovOverlap>* pFovList = &_fovFovOverlapLists[i][iTrig][iCam];
				for(list<FovFovOverlap>::iterator ite=pFovList->begin(); ite!=pFovList->end(); ite++)
				{
					if(IsFovFovOverlapForIllums(&(*ite), piIllumIndices, iNumIllums))
					{
						iFovFovCount++;
					}
				}

				// CadFov
				iCadFovCount += (unsigned int)_cadFovOverlapLists[i][iTrig][iCam].size();

				//FidFov
				iFidFovCount += (unsigned int)_fidFovOverlapLists[i][iTrig][iCam].size();
			}
		}
	}

	iFovFovCount /=2;

	// Double check 3*3
	unsigned int iSum = 3*3*iFovFovCount+iCadFovCount+iFidFovCount;
	return(iSum);
}

// Check wether FovFov overlap's all mosaic image is in the list
// 
bool OverlapManager::IsFovFovOverlapForIllums(FovFovOverlap* pOverlap, unsigned int* piIllumIndices, unsigned int iNumIllums) const
{
	unsigned int iIndex1 = pOverlap->GetFirstMosaicImage()->Index();
	unsigned int iIndex2 = pOverlap->GetSecondMosaicImage()->Index();
	
	bool bFlag1=false, bFlag2=false;
	unsigned int i;
	for(i=0; i<iNumIllums; i++)
	{
		if(iIndex1 == piIllumIndices[i])
			bFlag1 = true;

		if(iIndex2 == piIllumIndices[i])
			bFlag2 = true;
	}

	return(bFlag1 && bFlag2);
}

// Just wait all threads finish current jobs, not kill threads and their manager
bool OverlapManager::FinishOverlaps()
{
	// Wait for all job threads to finish...
	_pJobManager->MarkAsFinished();
	while(_pJobManager->TotalJobs() > 0)
		Sleep(10);

	return true;
}

#pragma endregion

#pragma region Alignment result check/verification

// Calculate parameters for a line
// y ~= m*x+b
// m=(xy_sum*num-x_sum*y_sum)/(xx_sum*num-x_sum*x_sum)
// b = (y_sum - m*x_sum)/num
// pdx, pdy and iNum: input x and y arrays and their size
// pm and pb: output m and b
bool CalLineParameter(
	const double* pdx, const double* pdy, unsigned int iNum,
	double* pm, double* pb)
{
	double x_sum=0, y_sum =0, xy_sum=0, xx_sum=0;
	for(unsigned int i=0; i<iNum; i++)
	{
		x_sum += pdx[i];
		y_sum += pdy[i];
		xy_sum += (pdx[i]*pdy[i]);
		xx_sum += (pdx[i]*pdx[i]);
	}

	*pm = (xy_sum*iNum - x_sum*y_sum)/(xx_sum*iNum - x_sum*x_sum);
	*pb = (y_sum - (*pm)*x_sum)/iNum;

	return(true);
}

// Calculate max inconsist offset based on estimation of line
// pdx, pdy and iNum: input x and y arrays and their size
// dMultiSdvTh: control parameter
// pOffsetFromLine: output, the offsets of ys from line
bool CalInconsistBasedOnLine(
	const double* pdx, const double* pdy, 
	unsigned int iNum, double dMultiSdvTh,
	double* pOffsetFromLine)
{
	// Not enough data for process
	if(iNum <= 2)
		return(false);

//** For 3 points
	if(iNum == 3)
	{
		double dSlope01 = 0;
		if(pdx[1] != pdx[0])
			dSlope01 = (pdy[1]-pdy[0])/(pdx[1]-pdx[0]);
		double dSlope12 = 0;
		if(pdx[2] != pdx[1])
			dSlope12 = (pdy[2]-pdy[1])/(pdx[2]-pdx[1]);
		double dSlope02 = 0;
		if(pdx[2] != pdx[0])
			dSlope02 = (pdy[2]-pdy[0])/(pdx[2]-pdx[0]);
		// line through points 0 and 1
		if((fabs(dSlope01) < fabs(dSlope12)) && (fabs(dSlope01) < fabs(dSlope02)))
		{	
			pOffsetFromLine[0] = 0;
			pOffsetFromLine[1] = 0;
			pOffsetFromLine[2] = pdy[2] - (dSlope01*(pdx[2]-pdx[0])+pdy[0]);
		}
		// line through points 1 and 2
		if((fabs(dSlope12) < fabs(dSlope01)) && (fabs(dSlope12) < fabs(dSlope02)))
		{	
			pOffsetFromLine[0] = pdy[0] - (dSlope12*(pdx[0]-pdx[1])+pdy[1]);
			pOffsetFromLine[1] = 0;
			pOffsetFromLine[2] = 0;
		}
		// line through points 0 and 2
		if((fabs(dSlope02) < fabs(dSlope01)) && (fabs(dSlope02) < fabs(dSlope12)))
		{	
			pOffsetFromLine[0] = 0;
			pOffsetFromLine[1] = pdy[1] - (dSlope02*(pdx[1]-pdx[0])+pdy[0]);
			pOffsetFromLine[2] = 0;
		}

		return(true);
	}

//** For 4 points or more
	// Calculat the SDV of X
	double dSumX=0, dSumSqX=0;
	for(unsigned int i=0; i<iNum; i++)
	{
		dSumX += pdx[i];
		dSumSqX += pdx[i]*pdx[i];
	}
	double dSdvX = sqrt(dSumSqX/iNum - (dSumX/iNum)*(dSumX/iNum));

	if(dSdvX>2e-3)	// If SDV of X is big, a line with angle
	{
		// All pixels used in line creation
		double m, b;
		CalLineParameter(pdx, pdy, iNum, &m, &b);

		// Calculate mean and sdv
		double* pdDis = new double[iNum];
		double dSdv_dis, dMean_dis=0, dMean_dissquare=0;
		for(unsigned int i=0; i<iNum; i++)
		{
			pdDis[i] = pdy[i] - (m*pdx[i]+b);
			dMean_dis += pdDis[i];
			dMean_dissquare += (pdDis[i]*pdDis[i]);
		}
		dMean_dissquare /= iNum;
		dMean_dis /= iNum;
		dSdv_dis = sqrt(dMean_dissquare - dMean_dis*dMean_dis);
		// When Sdv is smaller, return true;
		if(dSdv_dis * sqrt((double)iNum) < 1)
		{
			for(unsigned int i=0; i<iNum; i++)
				pOffsetFromLine[i] = 0;

			delete [] pdDis;
			return(true);
		}

		// Ignore pixels with bigger distance from line based on mean and sdv
		double* pdTempx = new double[iNum];
		double* pdTempy = new double[iNum];
		unsigned int iCount=0;
		double dTh = dSdv_dis*dMultiSdvTh;
		for(unsigned int i=0; i<iNum; i++)
		{
			if(fabs(pdDis[i]-dMean_dis) < dTh)
			{
				pdTempx[iCount] = pdx[i];
				pdTempy[iCount] = pdy[i];
				iCount++;
			}
		}

		// Check sdv on X
		double dSumX=0, dSumSqX=0;
		for(unsigned int i=0; i<iCount; i++)
		{
			dSumX += pdTempx[i];
			dSumSqX += pdTempx[i]*pdTempx[i];
		}
		double dSdvX = sqrt(dSumSqX/iCount - (dSumX/iCount)*(dSumX/iCount));

		// Create new line only when Sdv on X is big enough
		if(dSdvX>2e-3)
			CalLineParameter(pdTempx, pdTempy, iCount, &m, &b);

		// Calculate the offset from line
		for(unsigned int i=0; i<iNum; i++)
		{
			pOffsetFromLine[i] = pdy[i] - (m*pdx[i] +b);
		}	
		delete [] pdDis;
		delete [] pdTempx;
		delete [] pdTempy;
	}
	else // If SDV of X is small, a flat line
	{
		// Mean  and sdv of y 
		double dSumY=0, dSumSqY=0;
		for(unsigned int i=0; i<iNum; i++)
		{
			dSumY += pdy[i];
			dSumSqY += pdy[i]*pdy[i];
		}
		double dMeanY = dSumY/iNum;
		double dSdvY = sqrt(dSumSqY/iNum-dMeanY*dMeanY);
		// When Sdv is smaller, return true;
		if(dSdvY * sqrt((double)iNum) < 1)
		{
			for(unsigned int i=0; i<iNum; i++)
				pOffsetFromLine[i] = 0;

			return(true);
		}
		
		// Ignore pixels with bigger distance from line based on mean and sdv
		double* pdTempy = new double[iNum];
		unsigned int iCount=0;
		double dTh = dSdvY*dMultiSdvTh;
		for(unsigned int i=0; i<iNum; i++)
		{
			if(fabs(pdy[i]-dMeanY) < dTh)
			{
				pdTempy[iCount] = pdy[i];
				iCount++;
			}
		}

		// Calculate the adjusted mean
		dSumY = 0;
		for(unsigned int i=0; i<iCount; i++)
		{
			dSumY += pdTempy[i];
		}
		dMeanY = dSumY/iCount;

		// Calculate the offset from line
		for(unsigned int i=0; i<iNum; i++)
		{
			pOffsetFromLine[i] = pdy[i] - dMeanY;
		}	

		delete [] pdTempy;
	}

	return(true);
}

// Check inconsist of Fov to Fov overlap reults for a panel
// piCoarseInconsistNum: output, number of carse alignmemt failed in check
// piFineInconsistNum: output, number of fine alignmemt failed in check
bool OverlapManager::FovFovAlignConsistCheckForPanel(int* piCoarseInconsistNum, int* piFineInconsistNum)
{
	*piCoarseInconsistNum = 0;
	*piFineInconsistNum = 0;
	int iNumLayer = (int)_pMosaicSet->GetNumMosaicLayers();
	for(int i=0; i<iNumLayer; i++)
	{
		for(int j=i; j<iNumLayer; j++)
		{
			int iCoarseNum, iFineNum;
			if(FovFovAlignConsistCheckForTwoIllum(i, j, &iCoarseNum, &iFineNum))
			{
				*piCoarseInconsistNum += iCoarseNum;
				*piFineInconsistNum += iFineNum;
			}
		}
	}

	return(true);
}

// Check inconsist of Fov to Fov overlap reults for layers/illuminations
// piCoarseInconsistNum: output, number of carse alignmemt failed in check
// piFineInconsistNum: output, number of fine alignmemt failed in check
bool OverlapManager::FovFovAlignConsistCheckForTwoIllum(
	unsigned int iLayer1, unsigned int iLayer2,
	int* piCoarseInconsistNum, int* piFineInconsistNum)
{
	*piCoarseInconsistNum = 0;
	*piFineInconsistNum = 0;
	int iNumTrig1 = _pMosaicSet->GetLayer(iLayer1)->GetNumberOfTriggers();
	int iNumTrig2 = _pMosaicSet->GetLayer(iLayer2)->GetNumberOfTriggers();

	for(int iTrig1=0; iTrig1<iNumTrig1; iTrig1++)	// trig1
	{
		int iTrig2Start = iTrig1-1;
		if(iTrig2Start < 0) iTrig2Start = 0;
		int iTrig2End = iTrig1+1;
		if(iTrig2End > iNumTrig2-1) iTrig2End = iNumTrig2-1;

		for(int iTrig2=iTrig2Start; iTrig2<=iTrig2End; iTrig2++) // trig2
		{
			/* for debug 
			if(iLayer1==0 && iTrig1==8 && iLayer2==0 && iTrig2==8)
			{
				int i= 0;
			}//*/

			int iCoarseNum, iFineNum;
			if(FovFovAlignConsistChekcForTwoTrig(
				iLayer1, iTrig1,
				iLayer2, iTrig2,
				&iCoarseNum, &iFineNum))
			{
				*piCoarseInconsistNum += iCoarseNum;
				*piFineInconsistNum += iFineNum;
			}
		}
	}

	return(true);
}

// Check FovFov overlap alignment results for two triggers
// piCoarseInconsistNum: output, number of carse alignmemt failed in check
// piFineInconsistNum: output, number of fine alignmemt failed in check
bool OverlapManager::FovFovAlignConsistChekcForTwoTrig(
	unsigned int iLayer1, unsigned int iTrig1,
	unsigned int iLayer2, unsigned int iTrig2,
	int* piCoarseInconsistNum, int* piFineInconsistNum)
{
	// Whether overlaps are for the same trigger
	// cam to cam overlap for the same trigger
	bool bSameTrig = false;
	if(iLayer1 == iLayer2 && iTrig1 == iTrig2)
		bSameTrig = true;

	int iNumCam = _pMosaicSet->GetLayer(iLayer1)->GetNumberOfCameras();
	
	// No enough informaiton for consistent check 
	if(iNumCam<=2 || (bSameTrig && iNumCam<=3))
		return(0);

	// Collect all overlaps for consistent check
	list<FovFovOverlap*> trigOverlapPtrList;
	for(int iCam=0; iCam<iNumCam; iCam++)
	{
		// Get FovFov overlap list for a Fov in first trig
		FovFovOverlapList* pList = &_fovFovOverlapLists[iLayer1][iTrig1][iCam];
		for(FovFovOverlapList::iterator i = pList->begin(); i != pList->end(); i++)
		{
			// If right overlap and processed
			bool bFlag = (i->IsFromIllumTrigs(iLayer1, iTrig1, iLayer2, iTrig2) && i->IsProcessed() && i->IsGoodForSolver());
			if(bFlag)
			{
				bool bInList = false;
				if(bSameTrig)	// If cam to cam overlap for the same trigger
				{
					// Whether the overlap is in the list
					for(list<FovFovOverlap*>::iterator j = trigOverlapPtrList.begin(); j != trigOverlapPtrList.end(); j++)
					{
						if(&(*i) == *j)
						{
							bInList = true;
							break;
						}
					}
				}
				
				// Add to consist check list
				if(!bInList) 
					trigOverlapPtrList.push_back(&(*i)); 
			}
		}

		if(!bSameTrig)
		{
			// Get FovFov overlap list for a Fov in second trig
			pList = &_fovFovOverlapLists[iLayer2][iTrig2][iCam];
			for(FovFovOverlapList::iterator i = pList->begin(); i != pList->end(); i++)
			{
				// If right overlap and processed
				bool bFlag = (i->IsFromIllumTrigs(iLayer1, iTrig1, iLayer2, iTrig2) && i->IsProcessed());
				if(bFlag)
				{
					trigOverlapPtrList.push_back(&(*i)); 
				}
			}
		}
	}

	// Coarse should in front of fine
	*piCoarseInconsistNum = FovFovCoarseInconsistCheck(&trigOverlapPtrList);
	*piFineInconsistNum = FovFovFineInconsistCheck(&trigOverlapPtrList);

	return(true);
}

// Check inconsist of coarse alignment results for Fov and Fov
// Should be called before FovFovFineInconsistCheck()
// All coarse alignment results failed in check will mark as no good for solver
// pList: list of the overlaps for check
// Return number of coarse alignment failed in check
int OverlapManager::FovFovCoarseInconsistCheck(list<FovFovOverlap*>* pList)
{
	if(pList->size()<= 2)
		return(0);

	// Collect all valid coarse pair
	list<FovFovOverlap*> validList;
	for(list<FovFovOverlap*>::iterator j = pList->begin(); j != pList->end(); j++)
	{
		// Coarse alignment results will affect fine alignment
		if((*j)->IsAdjustedBasedOnCoarseAlignment())	
			validList.push_back((*j));
	}
	int iNum = (int)validList.size();
	if(iNum<=2) return(0);

	bool bFromSameDevice = (*pList->begin())->IsFromSameDevice();

	// Collect data for consistent check
	double* pdRowOffsets = new double[iNum];
	double* pdColOffsets = new double[iNum];
	double* pdNorminalX = new double[iNum];
	double* pdNorminalY = new double[iNum];
	int iCount = 0;
	double dx, dy;
	for(list<FovFovOverlap*>::iterator j = validList.begin(); j != validList.end(); j++)
	{
		(*j)->GetCoarsePair()->NorminalCenterInWorld(&dx, &dy);
		pdNorminalX[iCount] = dx;
		pdNorminalY[iCount] = dy;
		pdRowOffsets[iCount] = (*j)->GetCoarsePair()->GetCorrelationResult().RowOffset;
		pdColOffsets[iCount] = (*j)->GetCoarsePair()->GetCorrelationResult().ColOffset;
		iCount++;
	}

	// Consistent check
	int iReturnFlag = 0;
	double* pColOffsetFromLine = new double[iNum];
	double* pRowOffsetFromLine = new double[iNum];
	double dMultiSdvTh = 1.1;
	// Column check
	double dMaxColInconsist = CorrelationParametersInst.dMaxColInconsistInPixel;
	if(!bFromSameDevice) 
		dMaxColInconsist += CorrelationParametersInst.dColAdjust4DiffDevice;
	bool bFlag = CalInconsistBasedOnLine(pdNorminalX, pdColOffsets, iNum, dMultiSdvTh, pColOffsetFromLine);
	if(bFlag)
	{
		iCount = 0;
		for(list<FovFovOverlap*>::iterator j = validList.begin(); j != validList.end(); j++)
		{
			if(fabs(pColOffsetFromLine[iCount]) > dMaxColInconsist)
			{
				(*j)->SetIsGoodForSolver(false);
				LOG.FireLogEntry(LogTypeDiagnostic, "OverlapManager::FovFovCoarseInconsistCheck(): InConsist detected in Column of overlap (Layer=%d, Trig=%d, Cam=%d) and (Layer=%d, Trig=%d, cam=%d)",
					(*j)->GetFirstMosaicImage()->Index(), (*j)->GetFirstTriggerIndex(), (*j)->GetFirstCameraIndex(),
					(*j)->GetSecondMosaicImage()->Index(), (*j)->GetSecondTriggerIndex(), (*j)->GetSecondCameraIndex());
				iReturnFlag++;

				// for debug
				//(*j)->DumpOvelapImages();
				//(*j)->DumpResultImages();
			}
			iCount++;
		}
	}

	// Row check 
	double dMaxRowInconsist = CorrelationParametersInst.dMaxRowInconsistInPixel;
	if(!bFromSameDevice) 
		dMaxRowInconsist += CorrelationParametersInst.dRowAdjust4DiffDevice;
	bFlag = CalInconsistBasedOnLine(pdNorminalY, pdRowOffsets, iNum, dMultiSdvTh, pRowOffsetFromLine);
	if(bFlag)
	{
		iCount = 0;
		for(list<FovFovOverlap*>::iterator j = validList.begin(); j != validList.end(); j++)
		{
			if(fabs(pRowOffsetFromLine[iCount]) > dMaxRowInconsist)
			{
				if((*j)->IsGoodForSolver()) // Make sure not count twice
				{
					(*j)->SetIsGoodForSolver(false);
					LOG.FireLogEntry(LogTypeDiagnostic, "OverlapManager::FovFovCoarseInconsistCheck():InConsist detected in Row of overlap (Layer=%d, Trig=%d, Cam=%d) and (Layer=%d, Trig=%d, cam=%d)",
						(*j)->GetFirstMosaicImage()->Index(), (*j)->GetFirstTriggerIndex(), (*j)->GetFirstCameraIndex(),
						(*j)->GetSecondMosaicImage()->Index(), (*j)->GetSecondTriggerIndex(), (*j)->GetSecondCameraIndex());
					iReturnFlag++;

					// for debug
					//(*j)->DumpOvelapImages();
					//(*j)->DumpResultImages();
				}
			}
			iCount++;
		}
	}

	// Clean up
	delete [] pColOffsetFromLine;
	delete [] pRowOffsetFromLine;
	delete [] pdRowOffsets;
	delete [] pdColOffsets;
	delete [] pdNorminalX;
	delete [] pdNorminalY;

	return(iReturnFlag);
}

// Check inconsist of fine alignment results for Fov and Fov
// Should be called after FovFovCoarseInconsistCheck()
// All fine alignment results failed in check will mark as no good for solver
// All overlaps with coarse alignment no good for solver will be skipped
// All fine alignments with zero weight in solver will be skipped
// pList: list of the overlaps for check
// Return number of fine alignment failed in check
int OverlapManager::FovFovFineInconsistCheck(list<FovFovOverlap*>* pList)
{
	list<double> rowOffsetList, colOffsetList, norminalXList, norminalYList;
	list<CorrelationPair*> pairList;

	// Collect data for consistent check
	int iCount = 0;
	double dx, dy;
	for(list<FovFovOverlap*>::iterator j = pList->begin(); j != pList->end(); j++)
	{
		// Skip overlap with coarse alignment no good for solver
		if(!(*j)->IsGoodForSolver()) continue; 

		// Coarse offset that affect fine alignment
		double dCoarseRowOffset = 0;
		double dCoarseColOffset = 0;
		if((*j)->IsAdjustedBasedOnCoarseAlignment())
		{
			dCoarseRowOffset = (*j)->GetCoarsePair()->GetCorrelationResult().RowOffset;
			dCoarseColOffset = (*j)->GetCoarsePair()->GetCorrelationResult().ColOffset;
		}

		list<CorrelationPair>* pairListPtr = (*j)->GetFinePairListPtr();
		for(list<CorrelationPair>::iterator k = pairListPtr->begin(); k != pairListPtr->end(); k++)
		{
			// If weight for solver is zero, skip 
			double dWeight = EquationWeights::Instance().CalWeight(&(*k));
			if(dWeight <= 0) continue;

			pairList.push_back(&(*k));

			k->NorminalCenterInWorld(&dx, &dy);
			norminalXList.push_back(dx);
			norminalYList.push_back(dy);
			rowOffsetList.push_back(k->GetCorrelationResult().RowOffset + dCoarseRowOffset);
			colOffsetList.push_back(k->GetCorrelationResult().ColOffset + dCoarseColOffset);
		}
	}

	int iNum = (int)norminalXList.size();
	
	// Not enough number for test 
	if(iNum<=2) return(0);

	bool bFromSameDevice = (*pList->begin())->IsFromSameDevice();

	double* pdRowOffsets = new double[iNum];
	double* pdColOffsets = new double[iNum];
	double* pdNorminalX = new double[iNum];
	double* pdNorminalY = new double[iNum];

	list<double>::iterator i1 = rowOffsetList.begin();
	list<double>::iterator i2 = colOffsetList.begin();
	list<double>::iterator i3 = norminalXList.begin();
	list<double>::iterator i4 = norminalYList.begin();
	for(int i=0; i<iNum; i++)
	{
		pdRowOffsets[i] = *i1;
		pdColOffsets[i] = *i2;
		pdNorminalX[i] = *i3;
		pdNorminalY[i] = *i4;

		i1++;
		i2++;
		i3++;
		i4++;
	}

	// Consistent check
	int iReturnFlag = 0;
	double* pColOffsetFromLine = new double[iNum];
	double* pRowOffsetFromLine = new double[iNum];
	double dMultiSdvTh = 1.1;
	// Column check
	double dMaxColInconsist = CorrelationParametersInst.dMaxColInconsistInPixel;
	if(!bFromSameDevice) 
		dMaxColInconsist += CorrelationParametersInst.dColAdjust4DiffDevice;
	CalInconsistBasedOnLine(pdNorminalX, pdColOffsets, iNum, dMultiSdvTh, pColOffsetFromLine);
	iCount = 0;
	for(list<CorrelationPair*>::iterator j =  pairList.begin(); j != pairList.end(); j++)
	{
		if(fabs(pColOffsetFromLine[iCount]) > dMaxColInconsist)
		{
			(*j)->SetIsGoodForSolver(false);
			FovFovOverlap* pOverlap = (FovFovOverlap*)(*j)->GetOverlapPtr();
			int iIndex = (*j)->GetIndex();
			LOG.FireLogEntry(LogTypeDiagnostic, "OverlapManager::FovFovFineInconsistCheck():InConsist detected in Column of overlap (Layer=%d, Trig=%d, Cam=%d) and (Layer=%d, Trig=%d, cam=%d), fine #%d",
				pOverlap->GetFirstMosaicImage()->Index(), pOverlap->GetFirstTriggerIndex(), pOverlap->GetFirstCameraIndex(),
				pOverlap->GetSecondMosaicImage()->Index(), pOverlap->GetSecondTriggerIndex(), pOverlap->GetSecondCameraIndex(),
				iIndex);
			iReturnFlag++;

			// for debug
			//pOverlap->DumpOvelapImages();
			//pOverlap->DumpResultImages();
		}

		iCount++;
	}

	// Row check 
	double dMaxRowInconsist = CorrelationParametersInst.dMaxRowInconsistInPixel;
	if(!bFromSameDevice) 
		dMaxRowInconsist += CorrelationParametersInst.dRowAdjust4DiffDevice;
	CalInconsistBasedOnLine(pdNorminalY, pdRowOffsets, iNum, dMultiSdvTh, pRowOffsetFromLine);
	iCount = 0;
	for(list<CorrelationPair*>::iterator j =  pairList.begin(); j != pairList.end(); j++)
	{
		if(fabs(pRowOffsetFromLine[iCount]) > dMaxRowInconsist)
		{
			if((*j)->IsGoodForSolver())	// Make sure not set it twice
			{
				(*j)->SetIsGoodForSolver(false);
				FovFovOverlap* pOverlap = (FovFovOverlap*)(*j)->GetOverlapPtr();
				int iIndex = (*j)->GetIndex();
				LOG.FireLogEntry(LogTypeDiagnostic, "OverlapManager::FovFovFineInconsistCheck():InConsist detected in Row of overlap (Layer=%d, Trig=%d, Cam=%d) and (Layer=%d, Trig=%d, cam=%d), fine #%d",
					pOverlap->GetFirstMosaicImage()->Index(), pOverlap->GetFirstTriggerIndex(), pOverlap->GetFirstCameraIndex(),
					pOverlap->GetSecondMosaicImage()->Index(), pOverlap->GetSecondTriggerIndex(), pOverlap->GetSecondCameraIndex(),
					iIndex);
				iReturnFlag++;

				// for debug
				//pOverlap->DumpOvelapImages();
				//pOverlap->DumpResultImages();
			}
		}
		
		iCount++;
	}

	// Clean up
	delete [] pColOffsetFromLine;
	delete [] pRowOffsetFromLine;
	delete [] pdRowOffsets;
	delete [] pdColOffsets;
	delete [] pdNorminalX;
	delete [] pdNorminalY;

	return(iReturnFlag);
}

#pragma endregion