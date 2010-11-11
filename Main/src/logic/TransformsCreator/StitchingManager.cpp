#include "StitchingManager.h"

#pragma region Constructor and initilization
StitchingManager::StitchingManager(OverlapManager* pOverlapManager)
{
	_pOverlapManager = pOverlapManager;

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
}

StitchingManager::~StitchingManager(void)
{
	if(_pSolver != NULL) delete _pSolver;
	if(_pMaskSolver != NULL) delete _pMaskSolver;
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
}

bool StitchingManager::IsReadyToCreateMask()
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






