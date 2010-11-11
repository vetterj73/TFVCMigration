#include "StitchingManager.h"

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

typedef pair<FovIndex, double> FovIndexMap;
typedef list<FovIndexMap> FovList;
bool operator<(const FovIndexMap& a, const FovIndexMap& b)
{
	return(a.second < b.second);
};

// Create a map between Fov and its order in solver
// piIllumIndices and iNumIllums: input, illuminations used by solver
// pOrderMap: output, the map between Fov and its order in solver
bool StitchingManager::CreateImageOrderInSolver(
	unsigned int* piIllumIndices, 
	unsigned iNumIllums,
	map<FovIndex, unsigned int>* pOrderMap)
{
	unsigned int i, iTrig;
	FovList fovList;
	FovList::iterator j, k;
	
	// Build trigger list, 
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
	unsigned int iCount = 0;
	for(j=fovList.begin(); j!=fovList.end(); j++)
	{
		unsigned int iIllumIndex = j->first.IlluminationIndex;
		unsigned int iTrigerINdex = j->first.TriggerIndex;
		MosaicImage* pMosaic = _pOverlapManager->GetMoaicImage(iIllumIndex);
		unsigned int iNumCams = pMosaic->NumCameras();
		for(i=0; i<iNumCams; i++)
		{
			FovIndex index(iIllumIndex, iTrig, i);
			pOrderMap->insert(pair<FovIndex, unsigned int>(index, iCount));
			iCount++;
		}
	}
		
	return(true);
}

bool StitchingManager::CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap)
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
}

bool StitchingManager::IsReadyToCreateMasks()
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

bool StitchingManager::IsReadyToCreateTransforms()
{
	for(unsigned int i=0; i<_pOverlapManager->NumIlluminations(); i++)
	{
		if(!_pOverlapManager->GetMoaicImage(i)->IsAcquisitionCompleted())
			return(false);
	}

	return(true);
}

bool StitchingManager::CreateMasks( )
{
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
				Image* img = pMosaic->GetImagePtr(iCam, iTrig);
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
				Image* img = pMosaic->GetMaskImagePtr(iCam, iTrig);
				
				UIRect rect(0,  img->Columns()-1, 0, img->Rows()-1);
				img->MorphFrom(_pPanelMaskImage, rect);
			}
		}
	}

	return(true);
}


bool StitchingManager::CreateTransforms()
{
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
				Image* img = pMosaic->GetImagePtr(iCam, iTrig);
				ImgTransform t = _pMaskSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
			}
		}
	}
}

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