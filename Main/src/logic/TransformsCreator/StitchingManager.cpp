#include "StitchingManager.h"


bool operator<(const FovIndex& a, const FovIndex& b)
{
	if(a.IlluminationIndex < b.IlluminationIndex)
		return (true);

	if(a.TriggerIndex < b.TriggerIndex)
		return(true);

	if(a.CameraIndex < b.CameraIndex)
		return(true);

	return(false);
}

bool operator>(const FovIndex& a, const FovIndex& b)
{
	if(a.IlluminationIndex > b.IlluminationIndex)
		return (true);

	if(a.TriggerIndex > b.TriggerIndex)
		return(true);

	if(a.CameraIndex > b.CameraIndex)
		return(true);

	return(false);
}

StitchingManager::StitchingManager(OverlapManager* pOvelapManager)
{
	_pOvelapManager = pOvelapManager;
}


StitchingManager::~StitchingManager(void)
{
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
		MosaicImage* pMosaic = _pOvelapManager->GetMoaicImage(iIllumIndex);
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
		MosaicImage* pMosaic = _pOvelapManager->GetMoaicImage(iIllumIndex);
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
	unsigned int iNumIllums = _pOvelapManager->NumIlluminations();
	unsigned int* piIllumIndices = new unsigned int[iNumIllums];

	for(unsigned int i=0; i<iNumIllums; i++)
		piIllumIndices[i] = i;

	return(CreateImageOrderInSolver(
		piIllumIndices, 
		iNumIllums,
		pOrderMap));

	delete [] piIllumIndices;
}



