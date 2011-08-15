#include "PanelAligner.h"
#include "OverlapManager.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"
#include "MorphJob.h"
#include "Panel.h"
#include <direct.h> //_mkdir

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

	_queueMutex = CreateMutex(0, FALSE, "PanelAlignMutex"); // Mutex is not owned
}

PanelAligner::~PanelAligner(void)
{
	CleanUp();

	CloseHandle(_queueMutex);
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

	_pOverlapManager = new OverlapManager(_pSet, pPanel, CorrelationParametersInst.NumThreads);
		
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
			&_maskMap, 
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

void PanelAligner::LogFiducialOverlaps(bool bLog)
{
	CorrelationParametersInst.bSaveFiducialOverlaps = bLog;
}

void PanelAligner::LogOverlaps(bool bLog)
{
	CorrelationParametersInst.bSaveOverlaps = bLog;
}

void PanelAligner::LogMaskVectors(bool bLog)
{
	CorrelationParametersInst.bSaveTransformVectors= bLog;
}

void PanelAligner::NumThreads(unsigned int numThreads)
{
	CorrelationParametersInst.NumThreads = numThreads;
}

void PanelAligner::FiducialSearchExpansionXInMeters(double fidSearchXInMeters)
{
	CorrelationParametersInst.dFiducialSearchExpansionX = fidSearchXInMeters;
}

void PanelAligner::FiducialSearchExpansionYInMeters(double fidSearchYInMeters)
{
	CorrelationParametersInst.dFiducialSearchExpansionY = fidSearchYInMeters;
}

void PanelAligner::UseCyberNgc4Fiducial()
{
	CorrelationParametersInst.fidSearchMethod = FIDCYBERNGC;
}

// Add single image (single entry protected by mutex)
bool PanelAligner::ImageAddedToMosaicCallback(
	unsigned int iLayerIndex, 
	unsigned int iTrigIndex, 
	unsigned int iCamIndex)
{
	// The image is suppose to enter one by one
	WaitForSingleObject(_queueMutex, INFINITE);

	//LOG.FireLogEntry(LogTypeSystem, "PanelAligner::AddImage():Fov Layer=%d Trig=%d Cam=%d added!", iLayerIndex, iTrigIndex, iCamIndex);

	_pOverlapManager->DoAlignmentForFov(iLayerIndex, iTrigIndex, iCamIndex);
	
	// Release mutex
	ReleaseMutex(_queueMutex);
	
	// Masks are created after the first layer is aligned...
	// The assumption being that masks are not needed for the first set...
	if(_iMaskCreationStage>0 && !_bMasksCreated)
	{
		if(iTrigIndex == 9 && iCamIndex == 5)
			iTrigIndex = 9;
		// Wait all current overlap jobs done, then create mask
		if(IsReadyToCreateMasks())
		{
			if(_pOverlapManager->FinishOverlaps())
			{
				if(CreateMasks())
					_bMasksCreated= true;
				else
				{
					//log fatal error
				}
			}
		}
	}

	// If we are all done with alignment, create the transforms...
	if(_pSet->HasAllImages() && _pOverlapManager->FinishOverlaps())
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
	
	_lastProcessedFids.clear();

	// Create matrix and vector for solver
	for(int i=0; i<_iMaskCreationStage; i++)
	{
		AddOverlapResultsForIllum(_pMaskSolver, i);
	}

	// Solve transforms
	_pMaskSolver->SolveXAlgHB();

	// Create job manager for mask morpho
	CyberJob::JobManager jm("MaskMorpho", CorrelationParametersInst.NumThreads);
	vector<MorphJob*> morphJobs;

	// For each mosaic image
	for(int i=0; i<_iMaskCreationStage; i++)
	{
		// Get calculated transforms
		MosaicLayer* pMosaic = _pSet->GetLayer(i);

		// Create content of mask images
		for(unsigned iTrig=0; iTrig<pMosaic->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->GetNumberOfCameras(); iCam++)
			{
				Image* maskImg = pMosaic->GetMaskImage(iCam, iTrig);				
				ImgTransform t = _pMaskSolver->GetResultTransform(i, iTrig, iCam);
				maskImg->SetTransform(t);
				
				//UIRect rect(0, 0, maskImg->Columns()-1, maskImg->Rows()-1);
				//maskImg->MorphFrom(_pOverlapManager->GetPanelMaskImage(), rect);

				MorphJob *pJob = new MorphJob(maskImg, _pOverlapManager->GetPanelMaskImage(),
					0, 0, maskImg->Columns()-1, maskImg->Rows()-1);
				jm.AddAJob((CyberJob::Job*)pJob);
				morphJobs.push_back(pJob);
			}
		}
	}

	// Wait until it is complete...
	jm.MarkAsFinished();
	while(jm.TotalJobs() > 0)
		Sleep(10);

	for(unsigned int i=0; i<morphJobs.size(); i++)
		delete morphJobs[i];
	morphJobs.clear();

	/*/ For Debug
	for(int i=0; i<_iMaskCreationStage; i++)
	{
		// Get calculated transforms
		MosaicLayer* pMosaic = _pSet->GetLayer(i);

		// Create content of mask images
		for(unsigned iTrig=0; iTrig<pMosaic->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->GetNumberOfCameras(); iCam++)
			{
				Image* maskImg = pMosaic->GetMaskImage(iCam, iTrig);				
				
				string s;
				char cTemp[100];
				sprintf_s(cTemp, 100, "%sMaskI%dT%dC%d.bmp", 
					CorrelationParametersInst.GetOverlapPath().c_str(),
					i, iTrig, iCam);
				s.append(cTemp);
				
				maskImg->Save(s);
				
			}
		}
	}//*/

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateMasks():Mask images are created");

	return(true);
}

// Create the transform for each Fov
bool PanelAligner::CreateTransforms()
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Begin to create transforms");
	int iNumIllums = _pSet->GetNumMosaicLayers();

	_lastProcessedFids.clear();
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

	_pOverlapManager->GetFidResultSetPoint()->LogResults();
	
	if(CorrelationParametersInst.bSaveTransformVectors)
	{
		mkdir(CorrelationParametersInst.sDiagnosticPath.c_str());
		char cTemp[255];
		string s;
		sprintf_s(cTemp, 100, "%sMaskVectorX.csv", CorrelationParametersInst.sDiagnosticPath.c_str()); 
		s.clear();
		s.assign(cTemp);
		_pSolver->OutputVectorXCSV(s);
	}
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
			FovFovOverlapList* pFovFovList =_pOverlapManager->GetFovFovListForFov(iIllumIndex, iTrig, iCam);
			for(FovFovOverlapListIterator ite = pFovFovList->begin(); ite != pFovFovList->end(); ite++)
			{
				if(ite->IsProcessed())
					solver->AddFovFovOvelapResults(&(*ite));
			}

			// Add Cad and Fov overlap results
			CadFovOverlapList* pCadFovList =_pOverlapManager->GetCadFovListForFov(iIllumIndex, iTrig, iCam);
			for(CadFovOverlapListIterator ite = pCadFovList->begin(); ite != pCadFovList->end(); ite++)
			{
				if(ite->IsProcessed())
					solver->AddCadFovOvelapResults(&(*ite));
			}

			// Add Fiducial and Fov overlap results
			FidFovOverlapList* pFidFovList =_pOverlapManager->GetFidFovListForFov(iIllumIndex, iTrig, iCam);
			for(FidFovOverlapListIterator ite = pFidFovList->begin(); ite != pFidFovList->end(); ite++)
			{
				if(ite->IsProcessed())
				{
					solver->AddFidFovOvelapResults(&(*ite));

					// These are used to verify that the last fids actually worked...
					_lastProcessedFids.push_back(*ite);
				}
			}
		}
	}
}

FidFovOverlapList* PanelAligner::GetLastProcessedFids()
{
	return &_lastProcessedFids;
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


