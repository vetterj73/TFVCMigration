#include "StitchingManager.h"
#include "Logger.h"

#pragma region Constructor and initilization
StitchingManager::StitchingManager(OverlapManager* pOverlapManager, Image* pPanelMaskImage)
{
	_pOverlapManager = pOverlapManager;
	_pPanelMaskImage = pPanelMaskImage;

	// Create solver for all illuminations
	bool bProjectiveTrans = false;
	CreateImageOrderInSolver(&_solverMap);	
	unsigned int iMaxNumCorrelation =  pOverlapManager->MaxCorrelations();  
	_pSolver = new RobustSolver(	
		&_solverMap, 
		iMaxNumCorrelation, 
		bProjectiveTrans);

	// Creat solver for mask creation if it is necessary
	_pMaskSolver = NULL;
	_iMaskCreationStage = _pOverlapManager->GetMaskCreationStage();
	if(_iMaskCreationStage >= 1)
	{
		unsigned int* piIllumIndices = new unsigned int[_iMaskCreationStage];
		for(int i=0; i<_iMaskCreationStage; i++)
		{
			piIllumIndices[i] = i;	
		}
		CreateImageOrderInSolver(piIllumIndices, _iMaskCreationStage, &_maskMap);
		delete [] piIllumIndices;
		iMaxNumCorrelation =  pOverlapManager->MaxMaskCorrelations();
		_pMaskSolver = new RobustSolver(
			&_solverMap, 
			iMaxNumCorrelation, 
			bProjectiveTrans);
	}

	_bMasksCreated = false;
}

StitchingManager::~StitchingManager(void)
{
	if(_pSolver != NULL) delete _pSolver;
	if(_pMaskSolver != NULL) delete _pMaskSolver;
}


void StitchingManager::Reset()
{
	_pOverlapManager->ResetforNewPanel();
	_pSolver->Reset();
	if(_pMaskSolver != NULL)
		_pMaskSolver->Reset();

	_bMasksCreated = false;
}

// For function CreateImageOrderInSolver()
typedef pair<FovIndex, double> TriggerOffsetPair;
typedef list<TriggerOffsetPair> FovList;
bool operator<(const TriggerOffsetPair& a, const TriggerOffsetPair& b)
{
	return(a.second < b.second);
};
// Create a map between Fov and its order in solver
// piIllumIndices and iNumIllums: input, illuminations used by solver
// pOrderMap: output, the map between Fov and its order in solver
bool StitchingManager::CreateImageOrderInSolver(
	unsigned int* piIllumIndices, 
	unsigned iNumIllums,
	map<FovIndex, unsigned int>* pOrderMap) const
{
	unsigned int i, iTrig;
	FovList fovList;
	FovList::iterator j;

	// Build trigger offset pair list, 
	for(i=0; i<iNumIllums; i++) // for each illuminaiton 
	{
		// Get trigger centers in X
		unsigned int iIllumIndex = piIllumIndices[i];
		MosaicImage* pMosaic = _pOverlapManager->GetMoaicImage(iIllumIndex);
		unsigned int iNumTrigs = pMosaic->NumTriggers();
		double* dCenX = new double[iNumTrigs];
		pMosaic->TriggerCentersInX(dCenX);

		for(iTrig = 0; iTrig<iNumTrigs; iTrig++) // for each trigger
		{
			// Add to the list 
			FovIndex index(iIllumIndex, iTrig, 0);
			fovList.push_back(pair<FovIndex, double>(index, dCenX[iTrig]));
		}

		delete [] dCenX;
	}

	//** Check point
	// Sort list in ascending ordering
	fovList.sort();

	// Build FOVIndexMap	
	FovList::reverse_iterator k;
	unsigned int iCount = 0;
	for(k=fovList.rbegin(); k!=fovList.rend(); k++)
	{
		unsigned int iIllumIndex = k->first.IlluminationIndex;
		unsigned int iTrigIndex = k->first.TriggerIndex;
		MosaicImage* pMosaic = _pOverlapManager->GetMoaicImage(iIllumIndex);
		unsigned int iNumCams = pMosaic->NumCameras();
		for(i=0; i<iNumCams; i++)
		{
			FovIndex index(iIllumIndex, iTrigIndex, i);
			(*pOrderMap)[index] = iCount;
			iCount++;
		}
	}
		
	return(true);
}

bool StitchingManager::CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap) const
{
	unsigned int iNumIllums = _pOverlapManager->NumIlluminations();
	unsigned int* piIllumIndices = new unsigned int[iNumIllums];

	for(unsigned int i=0; i<iNumIllums; i++)
		piIllumIndices[i] = i;

	bool bFlag = CreateImageOrderInSolver(
		piIllumIndices, 
		iNumIllums,
		pOrderMap);

	delete [] piIllumIndices;

	return(bFlag);
}

#pragma endregion 

#pragma region create transforms
// Add an image buffer to a Fov
// pcBuf: input, image buffer
// iIllumIndex, iTrigIndex and iCamIndex: indics for a Fov
bool StitchingManager::AddOneImageBuffer(
	unsigned char* pcBuf,
	unsigned int iIllumIndex, 
	unsigned int iTrigIndex, 
	unsigned int iCamIndex)
{
	_pOverlapManager->GetMoaicImage(iIllumIndex)->AddImageBuffer(pcBuf, iCamIndex, iTrigIndex);
	_pOverlapManager->DoAlignmentForFov(iIllumIndex, iTrigIndex, iCamIndex);

	// Need create masks and masks havn't created
	if(_iMaskCreationStage>0 && !_bMasksCreated)
	{
		if(IsReadyToCreateMasks())
		{
			bool bFlag = CreateMasks();
			if(bFlag)
				_bMasksCreated= true;
			else
			{
				//log fatal error
			}
		}
	}

	if(IsReadyToCreateTransforms())
		CreateTransforms();

	return(true);
}

// Flag for create Masks
bool StitchingManager::IsReadyToCreateMasks() const
{
	if(_iMaskCreationStage <=0 )
		return(false);

	for(int i=0; i<_iMaskCreationStage; i++)
	{
		if(!_pOverlapManager->GetMoaicImage(i)->IsAcquisitionCompleted())
			return(false);
	}

	return(true);
}

// Flags for create transform for each Fov
bool StitchingManager::IsReadyToCreateTransforms() const
{
	for(unsigned int i=0; i<_pOverlapManager->NumIlluminations(); i++)
	{
		if(!_pOverlapManager->GetMoaicImage(i)->IsAcquisitionCompleted())
			return(false);
	}

	return(true);
}

// Creat Masks
bool StitchingManager::CreateMasks()
{
	LOG.FireLogEntry(LogTypeSystem, "StitchingManager::CreateMasks():begin to create mask");
	// Create matrix and vector for solver
	for(int i=0; i<_iMaskCreationStage; i++)
	{
		AddOverlapResultsForIllum(_pMaskSolver, i);
	}

	// Solve transforms
	_pMaskSolver->SolveXAlgHB();

	// For each mosaic image
	for(int i=0; i<_iMaskCreationStage; i++)
	{
		// Get calculated transforms
		MosaicImage* pMosaic = _pOverlapManager->GetMoaicImage(i);
		for(unsigned iTrig=0; iTrig<pMosaic->NumTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->NumCameras(); iCam++)
			{
				Image* img = pMosaic->GetImage(iCam, iTrig);
				ImgTransform t = _pMaskSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
			}
		}

		// Prepare mask images
		pMosaic->PrepareMaskImages();

		// Create content of mask images
		for(unsigned iTrig=0; iTrig<pMosaic->NumTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->NumCameras(); iCam++)
			{
				Image* img = pMosaic->GetMaskImage(iCam, iTrig);
				
				UIRect rect(0, 0, img->Columns()-1,  img->Rows()-1);
				img->MorphFrom(_pPanelMaskImage, rect);
			}
		}
	}

	LOG.FireLogEntry(LogTypeSystem, "StitchingManager::CreateMasks():Mask images are created");

	if(CorrParams.bSaveStitchedImage)
	{
		SaveStitchingImages("C:\\Temp\\AfterMask", _iMaskCreationStage);
		_pMaskSolver->OutputVectorXCSV("C:\\Temp\\MaskVectorX.csv");
	}

	return(true);
}

// Create the transform for each Fov
bool StitchingManager::CreateTransforms()
{
	LOG.FireLogEntry(LogTypeSystem, "StitchingManager::CreateTransforms():Begin to create transforms");
	int iNumIllums = _pOverlapManager->NumIlluminations();

	// Create matrix and vector for solver
	for(int i=0; i<iNumIllums; i++)
	{
		AddOverlapResultsForIllum(_pSolver, i);
	}

	// Solve transforms
	_pSolver->SolveXAlgHB();

	// For each mosaic image
	for(int i=0; i<iNumIllums; i++)
	{
		// Get calculated transforms
		MosaicImage* pMosaic = _pOverlapManager->GetMoaicImage(i);
		for(unsigned iTrig=0; iTrig<pMosaic->NumTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->NumCameras(); iCam++)
			{
				Image* img = pMosaic->GetImage(iCam, iTrig);
				ImgTransform t = _pSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
			}
		}
	}

	LOG.FireLogEntry(LogTypeSystem, "StitchingManager::CreateTransforms():Transforms are created");

	if(CorrParams.bSaveStitchedImage)
	{
		SaveStitchingImages("C:\\Temp\\Aligned", _iMaskCreationStage);
		_pSolver->OutputVectorXCSV("C:\\Temp\\AlignedVectorX.csv");
	}

	return(true);
}

// Add overlap results for a certain illumation/mosaic image to solver
void StitchingManager::AddOverlapResultsForIllum(RobustSolver* solver, unsigned int iIllumIndex)
{
	MosaicImage* pMosaic = _pOverlapManager->GetMoaicImage(iIllumIndex);
	for(unsigned iTrig=0; iTrig<pMosaic->NumTriggers(); iTrig++)
	{
		for(unsigned iCam=0; iCam<pMosaic->NumCameras(); iCam++)
		{
			// For certain Fov
			// Add calibration constraints
			solver->AddCalibationConstraints(pMosaic, iCam, iTrig);

			// Add Fov and Fov overlap results
			list<FovFovOverlap>* pFovFovList =_pOverlapManager->GetFovFovListForFov(iIllumIndex, iTrig, iCam);
			for(list<FovFovOverlap>::iterator ite = pFovFovList->begin(); ite != pFovFovList->end(); ite++)
			{
				if(ite->IsProcessed())
					solver->AddFovFovOvelapResults(&(*ite));
			}

			// Add Cad and Fov overlap results
			list<CadFovOverlap>* pCadFovList =_pOverlapManager->GetCadFovListForFov(iIllumIndex, iTrig, iCam);
			for(list<CadFovOverlap>::iterator ite = pCadFovList->begin(); ite != pCadFovList->end(); ite++)
			{
				if(ite->IsProcessed())
					solver->AddCadFovOvelapResults(&(*ite));
			}

			// Add Fiducial and Fov overlap results
			list<FidFovOverlap>* pFidFovList =_pOverlapManager->GetFidFovListForFov(iIllumIndex, iTrig, iCam);
			for(list<FidFovOverlap>::iterator ite = pFidFovList->begin(); ite != pFidFovList->end(); ite++)
			{
				if(ite->IsProcessed())
					solver->AddFidFovOvelapResults(&(*ite));
			}
		}
	}
}
#pragma endregion

#pragma region Create panel image

// Create a panel image based on a mosaic image 
// iIllumIndex: mosaic image index
// pPanelImage: output, the stitched image
void StitchingManager::CreateStitchingImage(unsigned int iIllumIndex, Image* pPanelImage) const
{
	MosaicImage* pMosaic = _pOverlapManager->GetMoaicImage(iIllumIndex);
	CreateStitchingImage(pMosaic, pPanelImage);
}

// This is a utility function
// Create a panel image based on a mosaic image
// pMosaic: input, mosaic image
// pPanelImage: output, the stitched image
void StitchingManager::CreateStitchingImage(const MosaicImage* pMosaic, Image* pPanelImage)
{
	// Trigger and camera centers in world space
	unsigned int iNumTrigs = pMosaic->NumTriggers();
	unsigned int iNumCams = pMosaic->NumCameras();
	double* pdCenX = new double[iNumTrigs];
	double* pdCenY = new double[iNumCams];
	pMosaic->TriggerCentersInX(pdCenX);
	pMosaic->CameraCentersInY(pdCenY);

	// Panel image Row bounds for Roi (decreasing order)
	unsigned int* piRectRows = new unsigned int[iNumTrigs+1];
	piRectRows[0] = pPanelImage->Rows();
	for(unsigned int i=1; i<iNumTrigs; i++)
	{
		double dX = (pdCenX[i-1] +pdCenX[i])/2;
		double dTempRow, dTempCol;
		pPanelImage->WorldToImage(dX, 0, &dTempRow, &dTempCol);
		piRectRows[i] = (unsigned int)dTempRow;
	}
	piRectRows[iNumTrigs] = 0; 

	// Panel image Column bounds for Roi (increasing order)
	unsigned int* piRectCols = new unsigned int[iNumCams+1];
	piRectCols[0] = 0;
	for(unsigned int i=1; i<iNumCams; i++)
	{
		double dY = (pdCenY[i-1] +pdCenY[i])/2;
		double dTempRow, dTempCol;
		pPanelImage->WorldToImage(0, dY, &dTempRow, &dTempCol);
		piRectCols[i] = (unsigned int)dTempCol;
	}
	piRectCols[iNumCams] = pPanelImage->Columns();

	// Morph each Fov to create stitched panel image
	for(unsigned int iTrig=0; iTrig<iNumTrigs; iTrig++)
	{
		for(unsigned int iCam=0; iCam<iNumCams; iCam++)
		{
			Image* pFov = pMosaic->GetImage(iCam, iTrig);
			UIRect rect(piRectCols[iCam], piRectRows[iTrig+1], piRectCols[iCam+1]-1,  piRectRows[iTrig]-1);
			pPanelImage->MorphFrom(pFov, rect);
		}
	}

	delete [] pdCenX;
	delete [] pdCenY;
	delete [] piRectRows;
	delete [] piRectCols;
}


//** for debug
void StitchingManager::SaveStitchingImages(string sName, unsigned int iNum)
{
	// Create image size
	double dPixelSize = 16.9e-6; 
	DRect rect = _pOverlapManager->GetValidRect();
	unsigned int iNumCols = (unsigned int)((rect.yMax - rect.yMin)/dPixelSize);
	unsigned int iNumRows = (unsigned int)((rect.xMax - rect.xMin)/dPixelSize);
	// create image transform
	double t[3][3];
	t[0][0] = dPixelSize;
	t[0][1] = 0;
	t[0][2] = rect.xMin;
	t[1][0] = 0;
	t[1][1] = dPixelSize;
	t[1][2] = rect.yMin;
	t[2][0] = 0;
	t[2][1] = 0;
	t[2][2] = 1;
	ImgTransform trans(t);
	//Create image
	bool bCreateOwnBuf = true;
	unsigned iBytePerpixel = 1;
	Image panelImage(iNumCols, iNumRows, iNumCols, iBytePerpixel, trans, trans, bCreateOwnBuf);

	// Create stitched images
	for(unsigned int i=0; i<iNum; i++)
	{
		CreateStitchingImage(i, &panelImage);
		char cTemp[100];
		sprintf_s(cTemp, 100, "%s_%d.bmp", sName, i); 
		string s;
		s.assign(cTemp);
		panelImage.Save(s);
	}
}

#pragma endregion