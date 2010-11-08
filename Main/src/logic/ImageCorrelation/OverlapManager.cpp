#include "OverlapManager.h"

OverlapManager::OverlapManager(
	MosaicImage* pMosaics, 
	CorrelationFlags** pFlags, 
	unsigned int iNumIllumination,
	Image* pCadImg, 
	DRect validRect)
{	
	_iMinOverlapSize = 100;
	
	_pMosaics = pMosaics;	
	_pFlags = pFlags;	
	_iNumIllumination = iNumIllumination;
	_pCadImg = pCadImg;
	_validRect = validRect;

	unsigned int i, j;
	_iNumCameras=0;
	_iNumTriggers=0;
	for(i=0; i<_iNumIllumination; i++)
	{
		if (_iNumCameras < pMosaics[i].NumCameras())
			_iNumCameras = pMosaics[i].NumCameras();

		if (_iNumTriggers < pMosaics[i].NumTriggers())
			_iNumTriggers = pMosaics[i].NumTriggers();
	}

	// Create 3D arrays for storage of overlaps
	_fovFovOverlapLists = new list<FovFovOverlap>**[_iNumIllumination];
	_cadFovOverlapLists = new list<CadFovOverlap>**[_iNumIllumination];
	_fidFovOverlapLists = new list<FidFovOverlap>**[_iNumIllumination];
	for(i=0; i<_iNumIllumination; i++)
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

	CreateFovFovOverlaps();
	CreateCadFovOverlaps();
}


OverlapManager::~OverlapManager(void)
{
	// Release 3D arrays for storage
	unsigned int i, j;
	for(i=0; i<_iNumIllumination; i++)
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
}

// Create Fov overlap for two illuminations
bool OverlapManager::CreateFovFovOverlapsForTwoIllum(unsigned int iIndex1, unsigned int iIndex2)
{
	CorrelationFlags flags = _pFlags[iIndex1][iIndex2];
	bool bCamCam = flags.GetCameraToCamera();
	bool bTrigTrig = flags.GetTriggerToTrigger();
	bool bMask = flags.GetMaskNeeded();
	bool bCad	= false; //need modify
	
	unsigned int iNumTrigs1 = _pMosaics[iIndex1].NumTriggers();
	unsigned int iNumCams1 = _pMosaics[iIndex1].NumCameras();
	double* pdCenX1 = new double[iNumTrigs1];
	double* pdCenY1 = new double[iNumCams1];
	_pMosaics[iIndex1].TriggerCentersInX(pdCenX1);
	_pMosaics[iIndex1].CameraCentersInY(pdCenY1);
	double dSpanTrig = (pdCenX1[iNumTrigs1-1] - pdCenX1[0])/(iNumTrigs1-1);
	double dSpanCam = (pdCenY1[iNumCams1-1] - pdCenY1[0])/(iNumCams1-1);
	
	unsigned int iNumTrigs2 = _pMosaics[iIndex2].NumTriggers();
	unsigned int iNumCams2 = _pMosaics[iIndex2].NumCameras();
	double* pdCenX2 = new double[iNumTrigs2];
	double* pdCenY2 = new double[iNumCams2];
	_pMosaics[iIndex2].TriggerCentersInX(pdCenX2);
	_pMosaics[iIndex2].CameraCentersInY(pdCenY2);

	unsigned int iCam1, iTrig1, iCam2, iTrig2;
	for(iTrig1 = 0; iTrig1<iNumTrigs1; iTrig1++)
	{
		for(iCam1 = 0; iCam1<iNumCams1; iCam1++)
		{
			// Cam to cam (Col to col) overlap
			if(bCamCam)
			{
				int iCamIndex1=-1, iCamIndex2=-1, iTrigIndex=-1;				
				// The nearest horizontal line1
				for(iTrig2=0; iTrig2<iNumTrigs2; iTrig2++)
				{	
					double dis = fabs(pdCenY1[iTrig1] -pdCenY2[iTrig2]);
					double dMinDis = 2*dSpanTrig;
					if(dMinDis > dis)
					{
						dMinDis = dis;
						iTrigIndex = iTrig2;
					}
				}
				// The nearest left image in horizontal line with a distance bigger than 0.5 span
				for(iCam2 = 0; iCam2<iNumCams2; iCam2++)
				{		
					double dis = pdCenX1[iCam1] -pdCenX2[iCam2];
					if(dis>dSpanCam*0.5)
						iCamIndex1 = iCam2;
				}
				// The nearest rightr image in horizontal line with a distance bigger than 0.5 span
				for(iCam2 = iNumCams2-1; iCam2>=0; iCam2++)
				{		
					double dis = pdCenX2[iCam2]-pdCenX1[iCam1];
					if(dis>dSpanCam*0.5)
						iCamIndex2 = iCam2;
				}

				// Create left overlap and add to list
				if(iCamIndex1>0 && iTrigIndex>0)
				{
					FovFovOverlap overlap(
						&_pMosaics[iIndex1], &_pMosaics[iIndex2],
						pair<unsigned int, unsigned int>(iCam1, iTrig1),
						pair<unsigned int, unsigned int>(iCamIndex1, iTrigIndex),
						_validRect, bMask);

					if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
					{
						_fovFovOverlapLists[iIndex1][iTrig1][iCam1].push_back(overlap);
						_fovFovOverlapLists[iIndex2][iTrigIndex][iCamIndex1].push_back(overlap);
					}
				}
				// Create right overlap and add to list
				if(iCamIndex2>0 && iTrigIndex>0)
				{
					FovFovOverlap overlap(
						&_pMosaics[iIndex1], &_pMosaics[iIndex2],
						pair<unsigned int, unsigned int>(iCam1, iTrig1),
						pair<unsigned int, unsigned int>(iCamIndex2, iTrigIndex),
						_validRect, bMask);

					if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
					{
						_fovFovOverlapLists[iIndex1][iTrig1][iCam1].push_back(overlap);
						_fovFovOverlapLists[iIndex2][iTrigIndex][iCamIndex2].push_back(overlap);
					}
				}
			} //if(bCamCam)

			// Trig to trig (Row to row) overlap
			if(bTrigTrig)
			{
				// Find the nearest vertical line
				int iCamIndex = -1;
				for(iCam2=0; iCam2<iNumCams2; iCam2++)
				{
					double dis = fabs(pdCenX1[iCam1] -pdCenX2[iCam2]);
					double dMinDis = 2*dSpanCam;
					if(dMinDis>dis && dis < 0.5*dSpanCam)
					{
						dMinDis = dis;
						iCamIndex = iCam2; 
					}
				}
				
				// Any image nearby in the nearest vertical line
				for(iTrig2=0; iTrig2<iNumTrigs2; iTrig2++)
				{
					double dis = fabs(pdCenX1[iCam1] -pdCenX2[iCam2]);
					if(dis < 0.8*dSpanTrig && iCamIndex>0)
					{
						FovFovOverlap overlap(
							&_pMosaics[iIndex1], &_pMosaics[iIndex2],
							pair<unsigned int, unsigned int>(iCam1, iTrig1),
							pair<unsigned int, unsigned int>(iCamIndex, iTrig2),
							_validRect, bMask);

						if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
						{
							_fovFovOverlapLists[iIndex1][iTrig1][iCam1].push_back(overlap);
							_fovFovOverlapLists[iIndex2][iTrig2][iCamIndex].push_back(overlap);
						}
					}
				}
			} // Trig to Trig
		}
	}

	return true;
}

// Create Fov and Fov overlaps
void OverlapManager::CreateFovFovOverlaps()
{
	unsigned int i, j;
	for(i=0; i<_iNumIllumination; i++)
	{
		for(j=i; j<_iNumIllumination; j++)
		{
			CreateFovFovOverlapsForTwoIllum(i, j);
		}
	}
}

// Create Cad and Fov overlaps
void OverlapManager::CreateCadFovOverlaps()
{
	unsigned int i, iCam, iTrig;
	for(i=0; i<_iNumIllumination; i++)
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

				if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
					_cadFovOverlapLists[i][iTrig][iCam].push_back(overlap);
			}
		}
	}
}

// Create Fiducial and Fov overlaps
void OverlapManager::CreateFidFovOverlaps()
{
}

// Reset mosaic images and overlaps for new panel inspection 
bool OverlapManager::ResetforNewPanel()
{
	// Reset all mosaic images for new panel inspection
	unsigned int i, iCam , iTrig;
	for(i=0; i<_iNumIllumination; i++)
	{
		_pMosaics[i].ResetForNextPanel();
	}

	// Reset all overlaps for new panel inspection
	for(i=0; i<_iNumIllumination; i++)
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
		if(i->IsValid())
			i->DoIt();
	}

	// Process valid Cad Fov overalp
	list<CadFovOverlap>* pCadFovList = &_cadFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]; 
	for(list<CadFovOverlap>::iterator i=pCadFovList->begin(); i!=pCadFovList->end(); i++)
	{
		if(i->IsValid())
			i->DoIt();
	}

	// Process valid fiducial Fov overlap
	list<FidFovOverlap>* pFidFovList = &_fidFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]; 
	for(list<FidFovOverlap>::iterator i=pFidFovList->begin(); i!=pFidFovList->end(); i++)
	{
		if(i->IsValid())
			i->DoIt();
	}

	return(true);
}

// Get FovFovOverlap list for certain Fov
list<FovFovOverlap>* OverlapManager::GetFovFovListForFov(
	unsigned int iMosaicIndex, 
	unsigned int iTrigIndex,
	unsigned int iCamIndex) const
{
	return(&_fovFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]);
}

// Get CadFovOverlap list for certain Fov
list<CadFovOverlap>* OverlapManager::GetCadFovListForFov(
	unsigned int iMosaicIndex, 
	unsigned int iTrigIndex,
	unsigned int iCamIndex) const
{
	return(&_cadFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]);
}

// Get FidFovOverlap list for certain Fov
list<FidFovOverlap>* OverlapManager::GetFidFovListForFov(
	unsigned int iMosaicIndex, 
	unsigned int iTrigIndex,
	unsigned int iCamIndex) const
{
	return(&_fidFovOverlapLists[iMosaicIndex][iTrigIndex][iCamIndex]);
}
