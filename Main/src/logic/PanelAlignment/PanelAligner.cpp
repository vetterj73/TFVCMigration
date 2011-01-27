#include "PanelAligner.h"
#include "OverlapManager.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"
#include "Panel.h"

void ImageAdded(int layerIndex, int cameraIndex, int triggerIndex, void* context)
{
	PanelAligner *pPanelAlign = (PanelAligner *)context;
	pPanelAlign->ImageAddedToMosaicCallback(layerIndex, triggerIndex, cameraIndex);
}

PanelAligner::PanelAligner(void)
{
	_pOverlapManager = NULL;
	_pSolver = NULL;
	_pMaskSolver = NULL;
}

PanelAligner::~PanelAligner(void)
{
	CleanUp();
}

// CleanUp internal stuff for new production or desctructor
void PanelAligner::CleanUp()
{
	if(_pOverlapManager != NULL)
		delete _pOverlapManager;

	if(_pSolver != NULL) 
		delete _pSolver;

	if(_pMaskSolver != NULL) 
		delete _pMaskSolver;

	_pOverlapManager = NULL;
	_pSolver = NULL;
	_pMaskSolver = NULL;


}

// Change production
bool PanelAligner::ChangeProduction(MosaicSet* pSet, Panel* pPanel)
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Begin panel change over");
	// CleanUp internal stuff for new production
	CleanUp();

	_pSet = pSet;
	_pSet->RegisterImageAddedCallback(ImageAdded, this);

	unsigned char* pCadBuf = pPanel->GetCadBuffer();
	unsigned char* pPanelMaskBuf = pPanel->GetMaskBuffer(); 

	_pOverlapManager = new OverlapManager(_pSet, pPanel);
		
	// Create solver for all illuminations
	bool bProjectiveTrans = false;
	CreateImageOrderInSolver(&_solverMap);	
	unsigned int iMaxNumCorrelation =  _pOverlapManager->MaxCorrelations();  
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
		iMaxNumCorrelation =  _pOverlapManager->MaxMaskCorrelations();
		_pMaskSolver = new RobustSolver(
			&_solverMap, 
			iMaxNumCorrelation, 
			bProjectiveTrans);
	}

	_bMasksCreated = false;
	_bResultsReady = false;

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Panel change over is done");
	return(true);
}

//Reset for next panel
void PanelAligner::ResetForNextPanel()
{
	_pOverlapManager->ResetforNewPanel();

	_pSolver->Reset();
	if(_pMaskSolver != NULL)
		_pMaskSolver->Reset();

	_bMasksCreated = false;
	_bResultsReady = false;

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ResetForNextPanel()");
}

// Add single image
bool PanelAligner::ImageAddedToMosaicCallback(
	unsigned int iLayerIndex, 
	unsigned int iTrigIndex, 
	unsigned int iCamIndex)
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::AddImage():Fov Layer=%d Trig=%d Cam=%d added!", iLayerIndex, iTrigIndex, iCamIndex);

	_pOverlapManager->DoAlignmentForFov(iLayerIndex, iTrigIndex, iCamIndex);

	// Masks are created after the first layer is aligned...
	// The assumption being that masks are not needed for the first set...
	if(_iMaskCreationStage>0 && !_bMasksCreated)
	{
		if(IsReadyToCreateMasks())
		{
			if(CreateMasks())
				_bMasksCreated= true;
			else
			{
				//log fatal error
			}
		}
	}

	// If we are all done with alignment, create the transforms...
	if(_pSet->HasAllImages())
	{
		CreateTransforms();
	}
	return(true);
}

// Flag for create Masks
bool PanelAligner::IsReadyToCreateMasks() const
{
	if(_iMaskCreationStage <= 0)
		return(false);

	for(int i=0; i<_iMaskCreationStage; i++)
	{
		if(!_pSet->GetLayer(i)->HasAllImages())
			return(false);
	}

	return(true);
}

// Creat Masks
bool PanelAligner::CreateMasks()
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateMasks():begin to create mask");
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
		MosaicLayer* pMosaic = _pSet->GetLayer(i);
		for(unsigned iTrig=0; iTrig<pMosaic->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->GetNumberOfCameras(); iCam++)
			{
				Image* img = pMosaic->GetImage(iCam, iTrig);
				ImgTransform t = _pMaskSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
			}
		}

		// Prepare mask images
		pMosaic->PrepareMaskImages();

		// Create content of mask images
		for(unsigned iTrig=0; iTrig<pMosaic->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->GetNumberOfCameras(); iCam++)
			{
				Image* img = pMosaic->GetMaskImage(iCam, iTrig);
				
				UIRect rect(0, 0, img->Columns()-1,  img->Rows()-1);
				img->MorphFrom(_pOverlapManager->GetPanelMaskImage(), rect);
			}
		}
	}

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateMasks():Mask images are created");

	return(true);
}

// Create the transform for each Fov
bool PanelAligner::CreateTransforms()
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Begin to create transforms");
	int iNumIllums = _pSet->GetNumMosaicLayers();

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
		MosaicLayer* pLayer = _pSet->GetLayer(i);
		for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
			{
				Image* img = pLayer->GetImage(iCam, iTrig);
				ImgTransform t = _pSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
			}
		}
	}

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Transforms are created");

	_bResultsReady = true;
	char cTemp[255];
	string s;
	sprintf_s(cTemp, 100, "%sMaskVectorX.csv", CorrParams.sStitchPath.c_str()); 
	s.clear();
	s.assign(cTemp);
	_pSolver->OutputVectorXCSV(s);

	return(true);
}

// Add overlap results for a certain illumation/mosaic image to solver
void PanelAligner::AddOverlapResultsForIllum(RobustSolver* solver, unsigned int iIllumIndex)
{
	MosaicLayer* pMosaic = _pSet->GetLayer(iIllumIndex);
	for(unsigned iTrig=0; iTrig<pMosaic->GetNumberOfTriggers(); iTrig++)
	{
		for(unsigned iCam=0; iCam<pMosaic->GetNumberOfCameras(); iCam++)
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
bool PanelAligner::CreateImageOrderInSolver(
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
		MosaicLayer* pMosaic = _pSet->GetLayer(iIllumIndex);
		unsigned int iNumTrigs = pMosaic->GetNumberOfTriggers();
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
		MosaicLayer* pMosaic = _pSet->GetLayer(iIllumIndex);
		unsigned int iNumCams = pMosaic->GetNumberOfCameras();
		for(i=0; i<iNumCams; i++)
		{
			FovIndex index(iIllumIndex, iTrigIndex, i);
			(*pOrderMap)[index] = iCount;
			iCount++;
		}
	}
		
	return(true);
}

bool PanelAligner::CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap) const
{
	unsigned int iNumIllums = _pSet->GetNumMosaicLayers();
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

bool PanelAligner::Save3ChannelImage(string filePath,
	unsigned char *pChannel1, unsigned char* pChannel2,	unsigned char* pChannel3, 
	int numColumns, int numRows)
{
	Bitmap *rbg = Bitmap::New3ChannelBitmap( 
		numRows, 
		numColumns, 
		pChannel1,
		pChannel2,
		pChannel3,
		numColumns,
		numColumns,
		numColumns);

	if(rbg == NULL)
		return false;

	rbg->write(filePath);
	delete rbg;
	return true;
}


