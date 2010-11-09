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

	_iMaskCreationStage = -1;

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
	// Correlation setting between two layers (illuminations)
	CorrelationFlags flags = _pFlags[iIndex1][iIndex2];
	bool bCamCam = flags.GetCameraToCamera();
	bool bTrigTrig = flags.GetTriggerToTrigger();
	bool bMask = flags.GetMaskNeeded();
	bool bCad	= false; //need modify
	
	// Camera centers in Y of world space and trigger centers in X of world space 
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

	// For each image in first layer (illuminaiton)
	unsigned int iCam1, iTrig1, iCam2, iTrig2;
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
				for(iTrig2=0; iTrig2<iNumTrigs2; iTrig2++)
				{	
					double dis = fabs(pdCenX1[iTrig1] -pdCenX2[iTrig2]);	
					if(dMinDis > dis)
					{
						dMinDis = dis;
						iTrigIndex = iTrig2;
					}		
				}
				
				if(iTrigIndex > 0)
				{
					// The nearest left image in the nearest trigger with a distance bigger than 0.5 span
					int iLeftCamIndex=-1;
					for(iCam2 = 0; iCam2<iNumCams2; iCam2++)
					{	// pdCenY1/2[i] increases with i	
						double dis = pdCenY1[iCam1] -pdCenY2[iCam2];
						if(dis > 0.5*dSpanCam && dis < 1.2*dSpanCam)
							iLeftCamIndex = iCam2;
					}
					// The nearest right image in the nearest trigger with a distance bigger than 0.5 span
					int iRightCamIndex=-1;
					for(iCam2 = iNumCams2-1; iCam2>=0; iCam2++)
					{	// pdCenY1/2[i] increases with i
						double dis = pdCenY2[iCam2]-pdCenY1[iCam1];
						if(dis > 0.5*dSpanCam  && dis < 1.2*dSpanCam)
							iRightCamIndex = iCam2;
					}

					// Create left overlap and add to list
					if(iLeftCamIndex>0)
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
					if(iRightCamIndex>0)
					{
						FovFovOverlap overlap(
							&_pMosaics[iIndex1], &_pMosaics[iIndex2],
							pair<unsigned int, unsigned int>(iCam1, iTrig1),
							pair<unsigned int, unsigned int>(iRightCamIndex, iTrigIndex),
							_validRect, bMask);

						if(overlap.Columns()>_iMinOverlapSize || overlap.Rows()>_iMinOverlapSize)
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
				double dMinDis = 1.0*dSpanCam; // If nearest camere is far away, ignore
				for(iCam2=0; iCam2<iNumCams2; iCam2++)
				{
					double dis = fabs(pdCenY1[iCam1] -pdCenY2[iCam2]);
					
					if(dMinDis>dis && dis < 0.5*dSpanCam)
					{
						dMinDis = dis;
						iCamIndex = iCam2; 
					}
				}
				
				if(iCamIndex>0)
				{
					// Any nearby trigger of the nearest camera 
					for(iTrig2=0; iTrig2<iNumTrigs2; iTrig2++)
					{
						double dis = fabs(pdCenX1[iTrig1] -pdCenX2[iTrig2]);
						if(dis < 1.1*dSpanTrig)
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

typedef list<pair<FovIndex, double>> FovList;

// Create a map between Fov and its order in solver
// piIllumIndices and iNumIllums: input, illuminations used by solver
// pOrderMap: output, the map between Fov and its order in solver
bool OverlapManager::CreateImageOrderInSolver(
	unsigned int* piIllumIndices, 
	unsigned iNumIllums,
	map<FovIndex, unsigned int>* pOrderMap)
{
	unsigned int i, iTrig;
	FovList fovList;
	FovList::iterator j, k;
	
	// Build trigger list, 
	// Trigger's center X increases in the list
	for(i=0; i<iNumIllums; i++) // for each illuminaiton 
	{
		// Get trigger centers in X
		unsigned int iIllumIndex = piIllumIndices[i];
		unsigned int iNumTrigs = _pMosaics[iIllumIndex].NumTriggers();
		double* dCenX = new double[iNumTrigs];
		_pMosaics[iIllumIndex].TriggerCentersInX(dCenX);

		for(iTrig = 0; iTrig<iNumTrigs; iTrig++) // for each trigger
		{
			// Find the right position
			bool bFlagInsert = false;
			for(j=fovList.begin(); j!=fovList.end(); j++)
			{
				if(j->second > dCenX[iTrig])
				{
					k = j;	// Insert position
					bFlagInsert = true;
					break;
				}
			}

			// Add to the list 
			FovIndex index(iIllumIndex, iTrig, 0);
			if(bFlagInsert) // Need check
			{	// Insert in a right position
				fovList.insert(k, pair<FovIndex, double>(index, dCenX[iTrig]));
			}
			else
			{// Add to the end of list if it is the biggest one or list is empty
				fovList.push_back(pair<FovIndex, double>(index, dCenX[iTrig]));
			}
		}

		delete [] dCenX;
	}

	// Build FOVIndexMap
	unsigned int iCount = 0;
	for(j=fovList.begin(); j!=fovList.end(); j++)
	{
		unsigned int iIllumIndex = j->first.IlluminationIndex;
		unsigned int iTrigerINdex = j->first.TriggerIndex;
		unsigned int iNumCams = _pMosaics[iIllumIndex].NumCameras();
		for(i=0; i<iNumCams; i++)
		{
			FovIndex index(iIllumIndex, iTrig, i);
			pOrderMap->insert(pair<FovIndex, unsigned int>(index, iCount));
			iCount++;
		}
	}
		
	return(true);
}

bool OverlapManager::CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap)
{
	unsigned int* piIllumIndices = new unsigned int[_iNumIllumination];

	for(unsigned int i=0; i<_iNumIllumination; i++)
		piIllumIndices[i] = i;

	return(CreateImageOrderInSolver(
		piIllumIndices, 
		_iNumIllumination,
		pOrderMap));

	delete [] piIllumIndices;
}

// Decide after what layer is completed, mask images need to be created
void OverlapManager::MaskCreationStage()
{
	unsigned int i, j;
	unsigned iMin = _iNumIllumination+10;
	for(i=0; i<_iNumIllumination; i++)
	{
		for(j=i; j<_iNumIllumination; j++)
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

	if(iMin > _iNumIllumination)
		_iMaskCreationStage = -1;
	else
		_iMaskCreationStage = iMin-1;
}


	
