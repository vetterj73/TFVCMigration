#include "OverlapManager.h"
#include "Logger.h"
#include "RenderShape.h"
#include "CorrelationParameters.h"

#pragma region constructor and reset
OverlapManager::OverlapManager(
	MosaicImage* pMosaics, 
	CorrelationFlags** pFlags, 
	unsigned int iNumIlluminations,
	Image* pCadImg, 
	double dCadImageResolution,
	Panel* pPanel)
{	
	// Inputs
	_pMosaics = pMosaics;	
	_pFlags = pFlags;	
	_iNumIlluminations = iNumIlluminations;
	_pCadImg = pCadImg;
	_pPanel = pPanel;
	_dCadImageResolution = dCadImageResolution;	

	// Valid panel area in world space 
	_validRect.xMin = 0;
	_validRect.yMin = 0;
	_validRect.xMax = _pPanel->xLength();
	_validRect.yMax = _pPanel->yLength();
	
	// Control parameter
	_iMinOverlapSize = CorrParams.iMinOverlapSize;

	// Calculate max number of cameras and triggers for mossiac images
	unsigned int i, j;
	_iNumCameras=0;
	_iNumTriggers=0;
	for(i=0; i<_iNumIlluminations; i++)
	{
		if (_iNumCameras < pMosaics[i].NumCameras())
			_iNumCameras = pMosaics[i].NumCameras();

		if (_iNumTriggers < pMosaics[i].NumTriggers())
			_iNumTriggers = pMosaics[i].NumTriggers();
	}

	// Create 3D arrays for storage of overlaps
	_fovFovOverlapLists = new list<FovFovOverlap>**[_iNumIlluminations];
	_cadFovOverlapLists = new list<CadFovOverlap>**[_iNumIlluminations];
	_fidFovOverlapLists = new list<FidFovOverlap>**[_iNumIlluminations];
	for(i=0; i<_iNumIlluminations; i++)
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

	//Create overlaps
	CreateFovFovOverlaps();
	CreateCadFovOverlaps();
	CreateFidFovOverlaps();

	// Decide the stage to calculate mask
	CalMaskCreationStage();
}

OverlapManager::~OverlapManager(void)
{
	// Release 3D arrays for storage
	unsigned int i, j;
	for(i=0; i<_iNumIlluminations; i++)
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

	if(_pFidImages != NULL)
		delete [] _pFidImages;
}

// Reset mosaic images and overlaps for new panel inspection 
bool OverlapManager::ResetforNewPanel()
{
	// Reset all mosaic images for new panel inspection
	unsigned int i, iCam , iTrig;
	for(i=0; i<_iNumIlluminations; i++)
	{
		_pMosaics[i].ResetForNextPanel();
	}

	// Reset all overlaps for new panel inspection
	for(i=0; i<_iNumIlluminations; i++)
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

#pragma endregion

#pragma region FovFovOverlaps
// Create Fov overlap for two illuminations
bool OverlapManager::CreateFovFovOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2)
{
	// Correlation setting between two layers (illuminations)
	CorrelationFlags flags = _pFlags[iIndex1][iIndex2];
	bool bCamCam = flags.GetCameraToCamera();
	bool bTrigTrig = flags.GetTriggerToTrigger();
	bool bMask = flags.GetMaskNeeded();
	bool bCad	= false; //need modify
	
	// Camera centers in Y of world space and trigger centers in X of world space 
	// Attention: Trgger center X is dcreaseing with trigger index
	// Cam center Y is increasing with camera index
	unsigned int iNumTrigs1 = _pMosaics[iIndex1].NumTriggers();
	unsigned int iNumCams1 = _pMosaics[iIndex1].NumCameras();
	double* pdCenX1 = new double[iNumTrigs1];
	double* pdCenY1 = new double[iNumCams1];
	_pMosaics[iIndex1].TriggerCentersInX(pdCenX1);
	_pMosaics[iIndex1].CameraCentersInY(pdCenY1);
	double dSpanTrig = (pdCenX1[0] - pdCenX1[iNumTrigs1-1])/(iNumTrigs1-1);
	double dSpanCam = (pdCenY1[iNumCams1-1] - pdCenY1[0])/(iNumCams1-1);
	
	unsigned int iNumTrigs2 = _pMosaics[iIndex2].NumTriggers();
	unsigned int iNumCams2 = _pMosaics[iIndex2].NumCameras();
	double* pdCenX2 = new double[iNumTrigs2];
	double* pdCenY2 = new double[iNumCams2];
	_pMosaics[iIndex2].TriggerCentersInX(pdCenX2);
	_pMosaics[iIndex2].CameraCentersInY(pdCenY2);

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
							&_pMosaics[iIndex1], &_pMosaics[iIndex2],
							pair<unsigned int, unsigned int>(iCam1, iTrig1),
							pair<unsigned int, unsigned int>(iLeftCamIndex, iTrigIndex),
							_validRect, bMask);

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
							&_pMosaics[iIndex1], &_pMosaics[iIndex2],
							pair<unsigned int, unsigned int>(iCam1, iTrig1),
							pair<unsigned int, unsigned int>(iRightCamIndex, iTrigIndex),
							_validRect, bMask);

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
								&_pMosaics[iIndex1], &_pMosaics[iIndex2],
								pair<unsigned int, unsigned int>(iCam1, iTrig1),
								pair<unsigned int, unsigned int>(iCamIndex, iTrig2),
								_validRect, bMask);

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
	for(i=0; i<_iNumIlluminations; i++)
	{
		for(j=i; j<_iNumIlluminations; j++)
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
	for(i=0; i<_iNumIlluminations; i++)
	{
		if(!_pMosaics[i].UseCad()) // If not use Cad
			continue;

		// If use Cad
		unsigned int iNumCameras = _pMosaics[i].NumCameras();
		unsigned int iNumTriggers = _pMosaics[i].NumTriggers();
		for(iTrig=0; iTrig<iNumTriggers; iTrig++)
		{
			for(iCam=0; iCam<iNumCameras; iCam++)
			{
				CadFovOverlap overlap(
					&_pMosaics[i],
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
bool OverlapManager::CreateFiducialImages()
{
	_pFidImages = NULL;
	unsigned int iNum = _pPanel->NumberOfFiducials();

	if(iNum <=0) 
	{
		LOG.FireLogEntry(LogTypeError, "OverlapManager::CreateFiducialImages():No fiducial avaliable for alignment");
		return(false);
	}
	_pFidImages = new Image[iNum];

	unsigned int iCount = 0;
	for(FeatureListIterator i = _pPanel->beginFiducials(); i != _pPanel->endFiducials(); i++)
	{
		double dScale = 1.0;
		RenderFiducial(
			&_pFidImages[iCount], 
			i->second, 
			_dCadImageResolution, 
			dScale);		
		
		// for Debug
		//string s;
		//char cTemp[100];
		//sprintf_s(cTemp, 100, "C:\\Temp\\Fid_%d.bmp", iCount);
		//s.append(cTemp);
		//_pFidImages[iCount].Save(s);

		iCount++;
	}

	return(true);
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
	double dExpandX = CorrParams.dFiducialSearchExpansionX;
	double dExpandY = CorrParams.dFiducialSearchExpansionY;
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
	unsigned int grayValue=255;
	int antiAlias=1;
	switch(pFid->GetShape())
	{
	case Feature::SHAPE_CROSS:
		RenderCross(*pImg, resolution, (CrossFeature*)pFid, grayValue, antiAlias);
		break;

	case Feature::SHAPE_DIAMOND:
		RenderDiamond(*pImg, resolution, (DiamondFeature*)pFid, grayValue, antiAlias);
		break;

	case Feature::SHAPE_DISC:
		RenderDisc(*pImg, resolution, (DiscFeature*)pFid, grayValue, antiAlias);
		break;

	case Feature::SHAPE_DONUT:
		RenderDonut(*pImg, resolution, (DonutFeature*)pFid, grayValue, antiAlias);
		break;

	case Feature::SHAPE_RECTANGLE:
		RenderRectangle(*pImg, resolution, (RectangularFeature*)pFid, grayValue, antiAlias);
		break;

	case Feature::SHAPE_TRIANGLE:
		RenderTriangle(*pImg, resolution, (TriangleFeature*)pFid, grayValue, antiAlias);
		break;

	case Feature::SHAPE_CYBER:
		RenderCyberShape(*pImg, resolution, (CyberFeature*)pFid, grayValue, antiAlias);
		break;

	default:
		LOG.FireLogEntry(LogTypeError, "OverlapManager::RenderFiducials() - unsupported fiducial type");	
	}
}

// Create Fiducial and Fov overlaps
void OverlapManager::CreateFidFovOverlaps()
{
	CreateFiducialImages();

	unsigned int i, iCam, iTrig;
	for(i=0; i<_iNumIlluminations; i++)
	{
		unsigned int iNumCameras = _pMosaics[i].NumCameras();
		unsigned int iNumTriggers = _pMosaics[i].NumTriggers();
		for(iTrig=0; iTrig<iNumTriggers; iTrig++)
		{
			for(iCam=0; iCam<iNumCameras; iCam++)
			{
				for(unsigned int iFid=0; iFid<_pPanel->NumberOfFiducials(); iFid++)
				{
					FidFovOverlap overlap(
						&_pMosaics[i],
						pair<unsigned int, unsigned int>(iCam, iTrig),
						&_pFidImages[iFid],
						_pFidImages[iFid].CenterX(),
						_pFidImages[iFid].CenterY(),
						_validRect);

					// Overlap validation check
					if(!overlap.IsValid())
						continue;

					// Overlap size check
					int iMinOverlapCols = _pFidImages[iFid].Columns() + 20-(int)(CorrParams.dFiducialSearchExpansionY/_dCadImageResolution);
					int iMinOverlapRows = _pFidImages[iFid].Rows() + 20-(int)(CorrParams.dFiducialSearchExpansionX/_dCadImageResolution);
					if((int)overlap.Columns()<iMinOverlapCols || (int)overlap.Rows()<iMinOverlapRows)
						continue;

					// Add overlap
					_fidFovOverlapLists[i][iTrig][iCam].push_back(overlap);
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
	// process valid FovFov Overlap  
	list<FovFovOverlap>* pFovFovList = &_fovFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex];
	for(list<FovFovOverlap>::iterator i=pFovFovList->begin(); i!=pFovFovList->end(); i++)
	{
		if(i->IsReadyToProcess())
		{
			i->DoIt();
			if(CorrParams.bSaveOverlap)
			{
				i->DumpOvelapImages();
				i->DumpResultImages();
			}
		}
	}

	// Process valid Cad Fov overalp
	list<CadFovOverlap>* pCadFovList = &_cadFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]; 
	for(list<CadFovOverlap>::iterator i=pCadFovList->begin(); i!=pCadFovList->end(); i++)
	{
		if(i->IsReadyToProcess())
		{
			i->DoIt();
			if(CorrParams.bSaveOverlap)
			{
				i->DumpOvelapImages();
				i->DumpResultImages();
			}
		}
	}

	// Process valid fiducial Fov overlap
	list<FidFovOverlap>* pFidFovList = &_fidFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]; 
	for(list<FidFovOverlap>::iterator i=pFidFovList->begin(); i!=pFidFovList->end(); i++)
	{
		if(i->IsReadyToProcess())
		{
			i->DoIt();
			if(CorrParams.bSaveOverlap)
			{
				i->DumpOvelapImages();
				i->DumpResultImages();
			}
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
	unsigned iMin = _iNumIlluminations+10;
	for(i=0; i<_iNumIlluminations; i++)
	{
		for(j=i; j<_iNumIlluminations; j++)
		{
			if(_pFlags[i][j].GetMaskNeeded())
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

	if(iMin > _iNumIlluminations)
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
	for(i=0; i<_iNumIlluminations; i++)
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
	unsigned int iSum = CorrParams.iFineMaxBlocksInRow * CorrParams.iFineMaxBlocksInCol * iFovFovCount+iCadFovCount+iFidFovCount;
	return(iSum);
}

//Report possible Maximum correlation will be used by solver to create mask
unsigned int OverlapManager::MaxMaskCorrelations() const
{
	if(_iMaskCreationStage<=0)
		return(0);

	unsigned int* piIllumIndices = new unsigned int[_iMaskCreationStage];

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
#pragma endregion
