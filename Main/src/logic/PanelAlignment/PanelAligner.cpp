#include "PanelAligner.h"
#include "OverlapManager.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"
#include "MorphJob.h"
#include "Panel.h"
#include "ColorImage.h"
#include <direct.h> //_mkdir
#include "EquationWeights.h"
#include "Bitmap.h"


#pragma region Constructor/Initialization/Call back

void ImageAdded(int layerIndex, int cameraIndex, int triggerIndex, void* context)
{
	PanelAligner *pPanelAlign = (PanelAligner *)context;
	pPanelAlign->ImageAddedToMosaicCallback(layerIndex, triggerIndex, cameraIndex);
}

void PanelAligner::RegisterAlignmentDoneCallback(ALIGNMENTDONE_CALLBACK pCallback, void* pContext)
{
	_registeredAlignmentDoneCallback = pCallback;
	_pCallbackContext = pContext;
}

void PanelAligner::UnregisterAlignmentDoneCallback()
{
	_registeredAlignmentDoneCallback = NULL;
	_pCallbackContext = NULL;
}

void PanelAligner::FireAlignmentDone(bool status)
{
	if(_registeredAlignmentDoneCallback != NULL)
		_registeredAlignmentDoneCallback(status);
}

LoggableObject* PanelAligner::GetLogger() 
{
	return &LOG;
}

MosaicSet* PanelAligner::GetMosaicSet() 
{
	return _pSet;
}

double PanelAligner::GetAlignmentTime() 
{
	return _dAlignmentTime;
}

// Add single image (single entry protected by mutex)
bool PanelAligner::ImageAddedToMosaicCallback(
	unsigned int iLayerIndex, 
	unsigned int iTrigIndex, 
	unsigned int iCamIndex)
{
	// The image is suppose to enter one by one
	//@todo - add a timeout
	WaitForSingleObject(_queueMutex, INFINITE);

	//LOG.FireLogEntry(LogTypeSystem, "PanelAligner::AddImage():Fov Layer=%d Trig=%d Cam=%d added!", iLayerIndex, iTrigIndex, iCamIndex);
	if(_iNumFovProced == 0)
		_StartTime = clock();

	OverlapAlignOption alignOption = COARSEFINE;
	if(CorrelationParametersInst.bUseTwoPassStitch && 
		(CorrelationParametersInst.bUseCameraModelStitch || 
		CorrelationParametersInst.bUseCameraModelIterativeStitch ) )
		alignOption = COARSEONLY;
	_pOverlapManager->DoAlignmentForFov(iLayerIndex, iTrigIndex, iCamIndex, alignOption);
	_iNumFovProced++;
	
	// Release mutex
	ReleaseMutex(_queueMutex);
	
	// If we are all done with alignment, create the transforms...
	if(_pSet->NumberOfImageTiles()==_iNumFovProced)
	{
		_pOverlapManager->FinishOverlaps();

		CreateTransforms();

		FireAlignmentDone(true);
	}

	return(true);
}

PanelAligner::PanelAligner(void)
{
	_pOverlapManager = NULL;
	_pSolver = NULL;

	_registeredAlignmentDoneCallback = NULL;
	_pCallbackContext = NULL;

	//_queueMutex = CreateMutex(0, FALSE, "PanelAlignMutex"); // Mutex is not owned
	_queueMutex = CreateMutex(0, FALSE, NULL); // Mutex is not owned

	_iNumFovProced = 0;
	CorrelationParametersInst.bCoarsePassDone = false;

	// for debug
	_iPanelCount = 0;

	// for QX
	_bOwnMosaicSetPanel = false;
	_pSet = NULL;
	_pPanel = NULL;
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

	_pOverlapManager = NULL;
	_pSolver = NULL;

	_iNumFovProced = 0;

	// for QX
	if(_bOwnMosaicSetPanel)
	{
		if(_pSet != NULL)
		{
			delete _pSet;
			_pSet = NULL;
		}

		if(_pPanel != NULL)
		{
			delete _pPanel;
			_pPanel = NULL;
		}
	}
}

// Change production
///<summary>
///Change Product, delete old solvers and create new solvers 
///</summary>
/// <param name="pSet"></param>
/// <param name="pPanel"></param>
/// Creates a new solver (of type slected in CorrelationParametersInst) for panel and (if needed) panel mask
bool PanelAligner::ChangeProduction(MosaicSet* pSet, Panel* pPanel)
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Begin panel change over");
	// CleanUp internal stuff for new production
	CleanUp();

	_pSet = pSet;
	_pPanel = pPanel;

	return ChangeProduction();
}
	
bool PanelAligner::ChangeProduction()
{
	_pSet->RegisterImageAddedCallback(ImageAdded, this) ;

	_pOverlapManager = new OverlapManager(_pSet, _pPanel, CorrelationParametersInst.NumThreads);
		
	// Create solver for all layers
	bool bProjectiveTrans = CorrelationParametersInst.bUseProjectiveTransform;
	bool bUseCameraModelStitch = CorrelationParametersInst.bUseCameraModelStitch;
	bool bUseCameraModelIterativeStitch = CorrelationParametersInst.bUseCameraModelIterativeStitch;
	CreateImageOrderInSolver(&_solverMap);	
	unsigned int iMaxNumCorrelations =  _pOverlapManager->MaxCorrelations();  
	//unsigned int iTotalNumberOfTriggers = _pSet->GetMosaicTotalNumberOfTriggers();
	if (bUseCameraModelIterativeStitch)
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():State of bUseCameraModelIterativeStitch True, %d", bUseCameraModelIterativeStitch);
		_pSolver = new RobustSolverIterative(	
						&_solverMap, 
						iMaxNumCorrelations,
						_pSet);  
	}
	else if (bUseCameraModelStitch)
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():State of bUseCameraModelStitch True, %d", bUseCameraModelStitch);
		_pSolver = new RobustSolverCM(	
						&_solverMap, 
						iMaxNumCorrelations,
						_pSet);  
	}
	else
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():State of bUseCameraModelStitch False, %d", bUseCameraModelStitch);
		_pSolver = new RobustSolverFOV(	
						&_solverMap, 
						iMaxNumCorrelations,
						_pSet,
						bProjectiveTrans);
	}

	if(_pSet->IsBayerPattern() && _pSet->IsSkipDemosaic() )
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Bayer pattern but skip demosaic!");

	// Creat solver for mask creation if it is necessary
	_bResultsReady = false;

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction(): End panel change over");
	return(true);
}




//Reset for next panel
void PanelAligner::ResetForNextPanel()
{
	_pOverlapManager->ResetforNewPanel();

	if(_pSet != NULL)
		_pSet->ClearAllImages();

	_pSolver->Reset();

	_bResultsReady = false;

	_iNumFovProced = 0;
}

#pragma endregion


#pragma region parameters set/get

void PanelAligner::LogFiducialOverlaps(bool bLog)
{
	CorrelationParametersInst.bSaveFiducialOverlaps = bLog;
}

void PanelAligner::LogOverlaps(bool bLog)
{
	CorrelationParametersInst.bSaveOverlaps = bLog;
}

void PanelAligner::LogTransformVectors(bool bLog)
{
	CorrelationParametersInst.bSaveTransformVectors= bLog;
}

void PanelAligner::LogPanelEdgeDebugImages(bool bLog)
{
	CorrelationParametersInst.bSavePanelEdgeDebugImages = bLog;
}

void PanelAligner::NumThreads(unsigned int numThreads)
{
	CorrelationParametersInst.NumThreads = numThreads;

	if(_pSet != NULL)
		_pSet->SetThreadNumber(numThreads);
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

void PanelAligner::UseProjectiveTransform(bool bValue)
{
	CorrelationParametersInst.bUseProjectiveTransform = bValue;
	// If the projective transform is used, panel may have serious warp, 
	// Disable fiducial alignment check, which highly depends on panel scale 
	if(CorrelationParametersInst.bUseProjectiveTransform) 
	{
		CorrelationParametersInst.bFiducialAlignCheck = false;
	}
}

void PanelAligner::UseCameraModelStitch(bool bValue)
{
	CorrelationParametersInst.bUseCameraModelStitch = bValue;
}
void PanelAligner::UseCameraModelIterativeStitch(bool bValue)
{
	CorrelationParametersInst.bUseCameraModelIterativeStitch = bValue;
}
void PanelAligner::SetUseTwoPassStitch(bool bValue)
{
	CorrelationParametersInst.bUseTwoPassStitch = bValue;
}

void PanelAligner::EnableFiducialAlignmentCheck(bool bValue)
{
	CorrelationParametersInst.bFiducialAlignCheck = bValue;
}

void PanelAligner::SetPanelEdgeDetection(
	bool bDetectPanelEdge, 
	int iLayerIndex4Edge,
	bool bConveyorLeft2Right,
	bool bConveyorFixedFrontRail)
{
	CorrelationParametersInst.bDetectPanelEdge = bDetectPanelEdge;
	CorrelationParametersInst.iLayerIndex4Edge = iLayerIndex4Edge;
	CorrelationParametersInst.bConveyorLeft2Right = bConveyorLeft2Right;
	CorrelationParametersInst.bConveyorFixedFrontRail = bConveyorFixedFrontRail;
}

void PanelAligner::SetCalibrationWeight(double dValue)
{
	EquationWeights::Instance().SetCalibrationScale(dValue);
}

void PanelAligner::SetSkipDemosaic(bool bValue)
{
	CorrelationParametersInst.bSkipDemosaic = bValue;
}

bool PanelAligner::GetCamModelPanelHeight(unsigned int iDeviceIndex, double pZCoef[16])
{
	if(!CorrelationParametersInst.bUseCameraModelStitch && !CorrelationParametersInst.bUseCameraModelIterativeStitch)
		return(false);

	return ((RobustSolverCM*)_pSolver)->GetPanelHeight(iDeviceIndex, pZCoef);
}

#pragma endregion


#pragma region create Mask


void PanelAligner::CalTransformsWithMask()
{
	clock_t sTime = clock();
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CalTransformsWithMask(): Begin solver on Mask");
	
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CalTransformsWithMask(): Begin Mask ovelap calculation");
	_pOverlapManager->AlignFovFovOverlapWithMask();
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CalTransformsWithMask(): End Mask ovelap calculation, time =%f", (float)(clock() - sTime)/CLOCKS_PER_SEC);

	// Consist check for FovFov alignment of each trigger
	if(CorrelationParametersInst.bFovFovAlignCheck)
	{
		bool bTrustCoarse = true;
		int iCoarseInconsistNum, iFineInconsistNum;
		_pOverlapManager->FovFovAlignConsistCheckForPanel(bTrustCoarse, &iCoarseInconsistNum, &iFineInconsistNum);
	}

	// Reset solver
	_pSolver->Reset();

	// Fill the solver
	bool bUseFiducials = true; 
	bool bPinPanelWithCalibration = false;
	bool bUseNominalTransform = false;
		// If no fiducial, pin the panel
	if(_pPanel->NumberOfFiducials() == 0)
	{
		bUseFiducials = false; 
		bPinPanelWithCalibration = true;
	}
	AddOverlapResults2Solver(
		_pSolver,
		bUseFiducials, 
		bPinPanelWithCalibration,
		bUseNominalTransform);

	// Solve transforms with panel leading edge but without fiducial information
	_pSolver->SolveXAlgH();
	//if camera model, must flatten fiducials
	_pSolver->FlattenFiducials( GetFidResultsSetPoint() );

	// Set transform to Fov images
	// For each mosaic image
	int iNumLayer = _pSet->GetNumMosaicLayers();
	for(int i=0; i<iNumLayer; i++)
	{
		// Get calculated transforms
		MosaicLayer* pLayer = _pSet->GetLayer(i);
		for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
			{
				Image* img = pLayer->GetImage(iTrig, iCam);
				ImgTransform t = _pSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
				img->CalInverseTransform();
			}
		}
	}

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CalTransformsWithMask(): End solver on Mask,  time =%f", (float)(clock() - sTime)/CLOCKS_PER_SEC);

	// Reset solver
	_pSolver->Reset();
}

#pragma endregion

#pragma region Align base on panel edge

// Align panel with panel leading edge information
bool PanelAligner::AlignWithPanelEdge(const EdgeInfo* pEdgeInfo, int iFidIndex)
{
	// Not use fiducial, not pin panel with calibration 
	// since panel leading edge will be used
	AddOverlapResults2Solver(_pSolver, false, false);

	// If fiducial information is available, edge x location is not needed 
	bool bSlopeOnly = false;
	if(iFidIndex>=0)
	{
		AddCurPanelFidOverlapResultsForPhyiscalFiducial(_pSolver, iFidIndex);
		bSlopeOnly = true;
	}

	// Add panel leading edge constraints
	MosaicLayer* pLayer = _pOverlapManager->GetMosaicSet()->GetLayer(pEdgeInfo->iLayerIndex);
	if(pEdgeInfo->type == LEFTONLYVALID || pEdgeInfo->type == BOTHVALID)
	{
		_pSolver->AddPanelEdgeContraints(
			pLayer, pEdgeInfo->iLeftCamIndex, pEdgeInfo->iTrigIndex, 
			pEdgeInfo->dLeftXOffset, pEdgeInfo->dPanelSlope, bSlopeOnly);
	}
	if(pEdgeInfo->type == RIGHTONLYVALID || pEdgeInfo->type == BOTHVALID)
	{
		_pSolver->AddPanelEdgeContraints(
			pLayer, pEdgeInfo->iRightCamIndex, pEdgeInfo->iTrigIndex, 
			pEdgeInfo->dRightXOffset, pEdgeInfo->dPanelSlope, bSlopeOnly);
	}

	// Solve transforms with panel leading edge but without fiducial information
	_pSolver->SolveXAlgH();

	// Get intermediate result transforms	
	int iNumLayer = _pSet->GetNumMosaicLayers();
	for(int i=0; i<iNumLayer; i++)
	{
		// Get calculated transforms
		pLayer = _pSet->GetLayer(i);
		for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
			{
				Image* img = pLayer->GetImage(iTrig, iCam);
				ImgTransform t = _pSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
			}
		}
	}
			
	// Reset solver
	_pSolver->Reset();

	// for debug
	if(CorrelationParametersInst.bSavePanelEdgeDebugImages)
	{
		// Get shift 100 pixel stitched image
		pLayer = _pSet->GetLayer(0);
		pLayer->SetXShift(true);
		Image* pTempImage = pLayer->GetGreyStitchedImage(); // For shifted stitched image
		pLayer->SetXShift(false);
		// Draw a 3-pixel width white line to represent leading edge
		unsigned char* pBuf = pTempImage->GetBuffer() + pTempImage->ByteRowStride()* (pTempImage->Rows()-1-100);
		::memset(pBuf, 255, pTempImage->ByteRowStride()*3);

		// Get shift 100 pixel Cad image
		unsigned int iNumRows = _pPanel->GetNumPixelsInX();
		unsigned int iNumCols = _pPanel->GetNumPixelsInY();
		unsigned char* pCadBuf =_pPanel->GetCadBuffer()+iNumCols*100;
		ImgTransform trans;
		Image tempImage2;	// For shifted Cad image
		tempImage2.Configure(iNumCols, iNumRows, iNumCols, trans, trans, true);
		::memset(tempImage2.GetBuffer(), 0, iNumCols*iNumRows);
		::memcpy(tempImage2.GetBuffer(), pCadBuf, iNumCols*(iNumRows-100)); 
		// Draw a 3-pixel width white line to represent leading edge
		pBuf = tempImage2.GetBuffer() + tempImage2.ByteRowStride()* (tempImage2.Rows()-1-100);
		::memset(pBuf, 255, tempImage2.ByteRowStride()*3);

		Bitmap* rbg = Bitmap::New2ChannelBitmap( 
			iNumRows, 
			iNumCols,
			pTempImage->GetBuffer(), 
			tempImage2.GetBuffer(),
			pTempImage->PixelRowStride(),
			tempImage2.PixelRowStride() );

		string sFileName;
		char cTemp[100];
		sprintf_s(cTemp, 100, "%sStitchedEdgeImage_%d_Fid#%d.bmp", CorrelationParametersInst.sDiagnosticPath.c_str(), _iPanelCount, iFidIndex);
		sFileName.append(cTemp);
		rbg->write(sFileName);
		delete rbg;
	}

	return(true);
}

// Use panel leading edge information to Align fiducials
bool PanelAligner::UseEdgeInfomation()
{
	// Get panel leading edge information
	EdgeInfo edgeInfo;
	bool bFlag = _pOverlapManager->GetEdgeDetector()->CalLeadingEdgeLocation(&edgeInfo);

	// If leading edge detection is failed or not processed
	if(edgeInfo.type == INVALID || edgeInfo.type == CONFLICTION || edgeInfo.type == NOPROCESSED) 
	{
		LOG.FireLogEntry(LogTypeError, "PanelAligner::CreateTransforms(): Panel leading edge detection failed with code %d!", (int)edgeInfo.type);
		return(false);
	}
		
	// If leading edge detection is success
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Panel leading edge detection success with code %d", (int)edgeInfo.type);

	// Align panel with panel leading edge information
	if(!AlignWithPanelEdge(&edgeInfo))
		return(false);
		
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Begin Fiducial search with edge!");
	// Create and Calculate fiducial overlaps for current panel
	bool bUseEdgeInfo = true;
	_pOverlapManager->DoAlignment4AllFiducial(bUseEdgeInfo);
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): End Fiducial search with edge!");

	// After all fiducial overlaps are calculated (It will clear old information automatically)
	_pOverlapManager->CreateFiducialResultSet(bUseEdgeInfo);
	double dConfidence = _pOverlapManager->GetFidResultsSetPoint()->CalConfidence();
	
	// If edge information is used but fiducial confidence is very low
	// If it is possible, using edge and good fiducial result further reduce search range
	if(dConfidence < 0.1)
	{
		if(!CorrelationParametersInst.bUseCameraModelStitch && !CorrelationParametersInst.bUseCameraModelIterativeStitch)
		{
			int iGoodIndex; 
			bFlag = _pOverlapManager->GetFidResultsSetPoint()->IsOneGoodOneAmbig(&iGoodIndex);
			if(bFlag)
			{
				if(!AlignWithPanelEdge(&edgeInfo, iGoodIndex))
					return(false);

				LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Begin Fiducial search with edge and fiducial #%d!", iGoodIndex);
				// Create and Calculate fiducial overlaps for current panel
				bool bHasEdgeFidInfo = true;
				_pOverlapManager->DoAlignment4AllFiducial(bUseEdgeInfo, bHasEdgeFidInfo);	
				LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): End Fiducial search with edge and fiducial #%d!", iGoodIndex);
				
				// After all fiducial overlaps are calculated (It will clear old information automatically)
				_pOverlapManager->CreateFiducialResultSet(bUseEdgeInfo);
				dConfidence = _pOverlapManager->GetFidResultsSetPoint()->CalConfidence();
			}
		}
	}

	// If edge information is used but fiducial confidence is very low
	// Fall back to without panel edge information
	if(dConfidence < 0.1)
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): With edge detection, Fiducial condidence is %d!", (int)(dConfidence*100));
			
		// not use edge information
		return(false);
	}

	return(true);
}

#pragma endregion

#pragma region create transforms

void PanelAligner::AddOverlapResults2Solver(
	RobustSolver* solver, 
	bool bUseFiducials, 
	bool bPinPanelWithCalibration,
	bool bUseNominalTransform)
{
	if(bUseFiducials)
		bPinPanelWithCalibration = false;

	// Add all loose constraints
	solver->AddAllLooseConstraints(	
		bPinPanelWithCalibration,
		bUseNominalTransform);

	// Add Fiducal and FOV, and CAD and Fov overlaps result
	int iNumLayer = _pSet->GetNumMosaicLayers();
	for(int iLayerIndex=0; iLayerIndex<iNumLayer; iLayerIndex++)
	{
		MosaicLayer* pLayer = _pSet->GetLayer(iLayerIndex);
		for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
			{
				// Add Cad and Fov overlap results
				CadFovOverlapList* pCadFovList =_pOverlapManager->GetCadFovListForFov(iLayerIndex, iTrig, iCam);
				for(CadFovOverlapListIterator ite = pCadFovList->begin(); ite != pCadFovList->end(); ite++)
				{
					if(ite->IsProcessed() && ite->IsGoodForSolver())
						solver->AddCadFovOvelapResults(&(*ite));
				}

				if(bUseFiducials)
				{
					// Add Fiducial and Fov overlap results
					FidFovOverlapList* pFidFovList =_pOverlapManager->GetFidFovListForFov(iLayerIndex, iTrig, iCam);
					for(FidFovOverlapListIterator ite = pFidFovList->begin(); ite != pFidFovList->end(); ite++)
					{
						if(ite->IsProcessed() && ite->IsGoodForSolver())
						{
							solver->AddFidFovOvelapResults(&(*ite));

							// These are used to verify that the last fids actually worked...
							_lastProcessedFids.push_back(*ite);
						}
					}
				}
			}
		}
	}

	// Add Fov and Fov overlap results
	FovFovOverlapList* pFovFovList =_pOverlapManager->GetFovFovOvelapSetPtr();
	for(FovFovOverlapListIterator ite = pFovFovList->begin(); ite != pFovFovList->end(); ite++)
	{
		if(ite->IsProcessed() && ite->IsGoodForSolver())
		solver->AddFovFovOvelapResults(&(*ite));
	}

	// Add Supplement overlaps
	AddSupplementOverlapResults(_pSolver);

	// Add input fiducial information
	if(_pSet->HasInputFidLocations())
	{
		map<int, FiducialLocation>* pFidLocMap = _pSet->GetInputFidLocMap();
		for(map<int, FiducialLocation>::iterator i=pFidLocMap->begin(); i!=pFidLocMap->end(); i++)
		{
			solver->AddInputFidLocations(&(i->second));
		}
	}
}


// Create the transform for each Fov
bool PanelAligner::CreateTransforms()
{	
	// If there are fiducial locations that need input from outside 
	if(_pSet->HasInputFidLocations())
	{
		int iSleepCount = 0;
		while(!_pSet->IsValidInputFidLocations())
		{
			if(iSleepCount > 100)
			{
				LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Waiting input fidcucial location time out");
				return(false);
			}
			Sleep(100);
			iSleepCount++;
		}
	}

	// for debug
	_iPanelCount++;
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Panel #%d is processing time = %f", _iPanelCount, (float)(clock() - _StartTime)/CLOCKS_PER_SEC);

	// Consist check for FovFov alignment of each trigger
	if(CorrelationParametersInst.bFovFovAlignCheck)
	{
		bool bTrustCoarse = false;
		int iCoarseInconsistNum, iFineInconsistNum;
		_pOverlapManager->FovFovAlignConsistCheckForPanel(bTrustCoarse, &iCoarseInconsistNum, &iFineInconsistNum);
	}

	// Must after consistent check and before transform calculation
	int iNumSupOverlap = _pOverlapManager->CalSupplementOverlaps();

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Begin to create transforms");
	int iNumLayer = _pSet->GetNumMosaicLayers();

	_lastProcessedFids.clear();

	// For debug
	// DisturbFiducialAlignment();

	// Alignment with panel leading edge and without fiducials 
	bool bUseEdgeInfo = false;
	if(CorrelationParametersInst.bDetectPanelEdge)
	{
		bUseEdgeInfo = UseEdgeInfomation();

		if(!bUseEdgeInfo)
		{
			// Do nominal fiducial overlap alignment
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Begin fallback Fiducial search without edge");
			_pOverlapManager->DoAlignment4AllFiducial(bUseEdgeInfo);
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): End fallback Fiducial search without edge");
		}

	}

	// After all fiducial overlaps are calculated
	_pOverlapManager->CreateFiducialResultSet(bUseEdgeInfo);

	if(!bUseEdgeInfo)
	{
		// Fiducial alignment check based on SIM calibration
		// Must after CreateFiducialResultSet()
		if(CorrelationParametersInst.bFiducialAlignCheck)
			FiducialAlignmentCheckOnCalibration();
	}

	// Pick the best alignment for each physical fiducial
	// Must after CreateFiducialResultSet()
	PickOneAlign4EachPanelFiducial();

	// Use nominal fiducail overlaps if edge info is not available
	bool bUseFiducials = !bUseEdgeInfo; 
	bool bPinPanelWithCalibration = false;
		// If no fiducail available, pin the panel
	if(_pPanel->NumberOfFiducials() == 0)
	{
		bUseFiducials = false; 
		bPinPanelWithCalibration = true;
		if (CorrelationParametersInst.bUseCameraModelIterativeStitch)
			((RobustSolverIterative*)_pSolver)->SetPinPanelWithCalibration(true);
	}
	AddOverlapResults2Solver(_pSolver, bUseFiducials, bPinPanelWithCalibration);
	
	// Use current panel fiducial overlaps if edge information is available
	if(bUseEdgeInfo)
		AddCurPanelFidOverlapResults(_pSolver);

	// Solve transforms
	_pSolver->SolveXAlgH();
	//if camera model, must flatten fiducials
	_pSolver->FlattenFiducials( GetFidResultsSetPoint() );
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Flatten Done, time = %f", (float)(clock() - _StartTime)/CLOCKS_PER_SEC);
	bool bMaskNeeded = _pOverlapManager->IsMaskNeeded();

	// For each mosaic image
	for(int i=0; i<iNumLayer; i++)
	{
		// Get calculated transforms
		MosaicLayer* pLayer = _pSet->GetLayer(i);
		for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
			{
				Image* img = pLayer->GetImage(iTrig, iCam);
				ImgTransform t = _pSolver->GetResultTransform(i, iTrig, iCam);
				img->SetTransform(t);
				if(!bMaskNeeded)
					img->CalInverseTransform();
			}
		}
	}

	// For fine pass (first run/second pass)
	if(CorrelationParametersInst.bUseTwoPassStitch && 
		(CorrelationParametersInst.bUseCameraModelStitch || CorrelationParametersInst.bUseCameraModelIterativeStitch ) )
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Start 2nd Pass Align, time = %f", (float)(clock() - _StartTime)/CLOCKS_PER_SEC);
		CorrelationParametersInst.bCoarsePassDone = true;
		// transforms from coarse align are loaded, ready to chop up fine aligns
		// calculate all fine overlaps
		/*FovFovOverlapList* pfovFovOverlapSet = _pOverlapManager->GetFovFovOvelapSetPtr();
		for(FovFovOverlapList::iterator i = pfovFovOverlapSet->begin(); i != pfovFovOverlapSet->end(); i++)
		{
			_pJobManager->AddAJob((CyberJob::Job*)*i);  // 
		}*/

		// Do fine alignment
		_pOverlapManager->AlignFovFovOverlap_FineOnly();

		_pSolver->Reset();
		_pOverlapManager->FinishOverlaps();
		AddOverlapResults2Solver(_pSolver, true);  // default to use fiducials instead of edge or calculated positions?
		// Solve transforms
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():2nd Pass Align call ALG_H, time = %f", (float)(clock() - _StartTime)/CLOCKS_PER_SEC);
		_pSolver->SolveXAlgH();

		// if two pass camera model (coarse then fine) 
		// we have only done a rough alignment, must now fix any problems in the coarse align and
		// chop for fine align
		//
		//@todo fix up any problems with the coarse align
		//
		// 

		//if camera model, must flatten fiducials
		_pSolver->FlattenFiducials( GetFidResultsSetPoint() );
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Flatten Done, time = %f", (float)(clock() - _StartTime)/CLOCKS_PER_SEC);
		// For each mosaic image
		for(int i=0; i<iNumLayer; i++)
		{
			// Get calculated transforms
			MosaicLayer* pLayer = _pSet->GetLayer(i);
			for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
			{
				for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
				{
					Image* img = pLayer->GetImage(iTrig, iCam);
					ImgTransform t = _pSolver->GetResultTransform(i, iTrig, iCam);
					img->SetTransform(t);
					if(!bMaskNeeded)
						img->CalInverseTransform();
				}
			}
		}
	}
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Transforms are created, time = %f", (float)(clock() - _StartTime)/CLOCKS_PER_SEC);

	// If mask is needed (second run)
	if(bMaskNeeded)
		CalTransformsWithMask();

	// Log fiducial confidence
	int iNumDevice = _pSet->GetNumDevice();
	if(iNumDevice > 1)
	{
		for(int i=0; i<iNumDevice; i++)
		{
			double dConfid =_pOverlapManager->GetFidResultsSetPoint()->CalConfidence(i);
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Device %d fiducial confidence = %d", i, (int)(dConfid*100) );
		}
	}
	double dConfid =_pOverlapManager->GetFidResultsSetPoint()->CalConfidence();
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Overall fiducial confidence = %d", (int)(dConfid*100) );

	_bResultsReady = true;

	// for debug
	//TestGetImagePatch();
	//TestSingleImagePatch();

	//_pOverlapManager->GetFidResultsSetPoint()->LogResults();
	
	if(CorrelationParametersInst.bSaveTransformVectors)
	{
		_mkdir(CorrelationParametersInst.sDiagnosticPath.c_str());
		char cTemp[255];
		string s;
		sprintf_s(cTemp, 100, "%sTransformVectorX_P%d.csv", CorrelationParametersInst.sDiagnosticPath.c_str(), _iPanelCount); 
		s.clear();
		s.assign(cTemp);
		_pSolver->OutputVectorXCSV(s);

		sprintf_s(cTemp, 100, "%sProjectiveTransform_P%d.csv", CorrelationParametersInst.sDiagnosticPath.c_str(), _iPanelCount);
		s.clear();
		s.assign(cTemp);
		OutputTransforms(s);

		sprintf_s(cTemp, 100, "%sFiducailResults_P%d.csv", CorrelationParametersInst.sDiagnosticPath.c_str(), _iPanelCount);
		s.clear();
		s.assign(cTemp);
		OutputFiducialForSolver(s);
	}

	_dAlignmentTime = (double)(clock() - _StartTime)/CLOCKS_PER_SEC;
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Overall alignment (no demosaic) time = %f",  _dAlignmentTime);

	return(true);
}

// Add current panel/(not nominal) fiducial overlap results
void PanelAligner::AddCurPanelFidOverlapResults(RobustSolver* solver)
{
	for(unsigned int k=0; k<_pPanel->NumberOfFiducials(); k++)
	{
		AddCurPanelFidOverlapResultsForPhyiscalFiducial(solver, k);
	}
}

void PanelAligner::AddCurPanelFidOverlapResultsForPhyiscalFiducial(RobustSolver* solver, int iIndex)
{
	FidFovOverlapList* pFidFovList =_pOverlapManager->GetCurPanelFidFovList4Fid(iIndex);
	for(FidFovOverlapListIterator ite = pFidFovList->begin(); ite != pFidFovList->end(); ite++)
	{
		if(ite->IsProcessed() && ite->IsGoodForSolver())
		{
			solver->AddFidFovOvelapResults(&(*ite));
			// These are used to verify that the last fids actually worked...
			_lastProcessedFids.push_back(*ite);
		}
	}
}

// Add Supplement overlap results to solver
void PanelAligner::AddSupplementOverlapResults(RobustSolver* solver)
{
	// Add Fov and Fov overlap results
	FovFovOverlapList* pFovFovList =_pOverlapManager->GetSupplementOverlaps();
	for(FovFovOverlapListIterator ite = pFovFovList->begin(); ite != pFovFovList->end(); ite++)
	{
		if(ite->IsProcessed() && ite->IsGoodForSolver())
			solver->AddFovFovOvelapResults(&(*ite));
	}
}

FidFovOverlapList* PanelAligner::GetLastProcessedFids()
{
	return &_lastProcessedFids;
}


PanelFiducialResultsSet* PanelAligner::GetFidResultsSetPoint() 
{
	return _pOverlapManager->GetFidResultsSetPoint();
};

// For function CreateImageOrderInSolver()
typedef pair<FovIndex, double> TriggerOffsetPair;
typedef list<TriggerOffsetPair> FovList;
bool operator<(const TriggerOffsetPair& a, const TriggerOffsetPair& b)
{
	return(a.second < b.second);
};

// Create a map between Fov and its order in solver
// piLayerIndices and iNumLayer: input, layers used by solver
// pOrderMap: output, the map between Fov and its order in solver
bool PanelAligner::CreateImageOrderInSolver(
	unsigned int* piLayerIndices, 
	unsigned iNumLayer,
	map<FovIndex, unsigned int>* pOrderMap) const
{
	unsigned int i, iTrig;
	FovList fovList;
	FovList::iterator j;
	unsigned int SolverTrigIndex(0);
	// Build trigger offset pair list, 
	for(i=0; i<iNumLayer; i++) // for each layer
	{
		// Get trigger centers in X
		unsigned int iLayerIndex = piLayerIndices[i];
		MosaicLayer* pLayer = _pSet->GetLayer(iLayerIndex);
		unsigned int iNumTrigs = pLayer->GetNumberOfTriggers();
		double* dCenX = new double[iNumTrigs];
		pLayer->TriggerCentersInX(dCenX);

		for(iTrig = 0; iTrig<iNumTrigs; iTrig++) // for each trigger
		{
			// Add to the list 
			FovIndex index(iLayerIndex, iTrig, 0);
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
		unsigned int iLayerIndex = k->first.LayerIndex;
		unsigned int iTrigIndex = k->first.TriggerIndex;
		MosaicLayer* pLayer = _pSet->GetLayer(iLayerIndex);
		list<SubSetCams> subTrigInfo = pLayer->GetSubTrigInfo();
		for(list<SubSetCams>::iterator iSub=subTrigInfo.begin(); iSub!=subTrigInfo.end(); iSub++)
		{
			for(i=iSub->iFirstCamIndex; i<=iSub->iLastCamIndex; i++)
			{
				FovIndex index(iLayerIndex, iTrigIndex, i);
				(*pOrderMap)[index] = iCount;
				if( !CorrelationParametersInst.bUseCameraModelStitch  && !CorrelationParametersInst.bUseCameraModelIterativeStitch ) 
					iCount++;
			}
			if( CorrelationParametersInst.bUseCameraModelStitch || CorrelationParametersInst.bUseCameraModelIterativeStitch ) 
				iCount++;
		}
	}
		
	return(true);
}

bool PanelAligner::CreateImageOrderInSolver(map<FovIndex, unsigned int>* pOrderMap) const
{
	unsigned int iNumLayer = _pSet->GetNumMosaicLayers();
	unsigned int* piLayerIndices = new unsigned int[iNumLayer];

	for(unsigned int i=0; i<iNumLayer; i++)
		piLayerIndices[i] = i;

	bool bFlag = CreateImageOrderInSolver(
		piLayerIndices, 
		iNumLayer,
		pOrderMap);

	delete [] piLayerIndices;

	return(bFlag);
}

// Check fiducial alignment based on SIM calibration
int PanelAligner::FiducialAlignmentCheckOnCalibration()
{
	// Create matrix and vector for solver without fiducial information	
	_pSolver->Reset();

	// Not use fiducial but pin panel with calibration
	AddOverlapResults2Solver(_pSolver, false, true);

	// Solve transforms without fiducial information
	_pSolver->SolveXAlgH();

	// Get the fiducial information
	PanelFiducialResultsSet* pFidResultsSet = GetFidResultsSetPoint();

	// Fiducial alignment check
	FiducialResultCheck fidChecker(pFidResultsSet, _pSolver);
	int iFlag = fidChecker.CheckFiducialResults();

	_pSolver->Reset();

	return(iFlag);
}

// Pick one alignment result of each panel/physical fiducial for solver
bool PanelAligner::PickOneAlign4EachPanelFiducial()
{
	// Get the fiducial information
	PanelFiducialResultsSet* pFidResultsSet = GetFidResultsSetPoint();
	for(int i=0; i<pFidResultsSet->Size(); i++)	// For each panel fiducial 
	{
		// Pick one alignment
		PanelFiducialResults* results = pFidResultsSet ->GetPanelFiducialResultsPtr(i);
		list<FidFovOverlap*>* resultList = results->GetFidOverlapListPtr();
		double dMaxWeight = -1;
		int iMaxIndex = -1;
		int iCount = 0;
		for(list<FidFovOverlap*>::iterator j = resultList->begin(); j != resultList->end(); j++)
		{
			if((*j)->IsProcessed() && (*j)->IsGoodForSolver() && (*j)->GetWeightForSolver()>0)
			{
				double dWeight = (*j)->GetWeightForSolver();
				if(dMaxWeight < dWeight)
				{
					dMaxWeight = dWeight;
					iMaxIndex = iCount;
				}
				iCount++;
			}
		}
		if(dMaxWeight<=0 || iMaxIndex<0)
			continue;

		// Set all but the selected one as not good for solver
		iCount = 0;
		for(list<FidFovOverlap*>::iterator j = resultList->begin(); j != resultList->end(); j++)
		{
			if((*j)->IsProcessed() && (*j)->IsGoodForSolver() && (*j)->GetWeightForSolver()>0)
			{
				if(iCount != iMaxIndex)
				{
					(*j)->SetIsGoodForSolver(false);
				}
				iCount++;
			}
		}
	}

	return(true);
}

#pragma endregion

#pragma region debug

// For debug
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

bool PanelAligner::Save3ChannelImage(string filePath,
	unsigned char *pChannel1, int iSpan1,
	unsigned char* pChannel2, int iSpan2,
	unsigned char* pChannel3, int iSpan3,
	int numColumns, int numRows)
{
	Bitmap *rbg = Bitmap::New3ChannelBitmap( 
		numRows, 
		numColumns, 
		pChannel1,
		pChannel2,
		pChannel3,
		iSpan1,
		iSpan2,
		iSpan3);

	if(rbg == NULL)
		return false;

	rbg->write(filePath);
	delete rbg;
	return true;
}

void PanelAligner::DisturbFiducialAlignment()
{
	unsigned int iNumLayer = _pSet->GetNumMosaicLayers();
	
	for(unsigned int iLayerIndex=0; iLayerIndex<iNumLayer; iLayerIndex++)
	{
		MosaicLayer* pLayer = _pSet->GetLayer(iLayerIndex);
		for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
			{
				// Add Fiducial and Fov overlap results
				FidFovOverlapList* pFidFovList =_pOverlapManager->GetFidFovListForFov(iLayerIndex, iTrig, iCam);
				for(FidFovOverlapListIterator ite = pFidFovList->begin(); ite != pFidFovList->end(); ite++)
				{
					// for debug
					if(iLayerIndex==2 && iTrig==0 && iCam==0)
					{
						// Simulate FOV location error
						CorrelationResult result = ite->GetCoarsePair()->GetCorrelationResult();
						result.RowOffset += 100;
						ite->GetCoarsePair()->SetCorrlelationResult(result);
					}
				}
			}
		}
	}
}

void PanelAligner::TestGetImagePatch()
{
	int iLayerIndex = 2;
	double dRes = _pSet->GetNominalPixelSizeX();

	MosaicDM::FOVPreferSelected setFov;
	//setFov.preferLR = MosaicDM::MosaicLayer::RIGHTFOV;
	//setFov.preferTB = MosaicDM::BOTTOMFOV;

	// Modify color for different FOV for debug purpose
	if(_pSet->IsBayerPattern() && !_pSet->IsSkipDemosaic())
	{
		MosaicLayer* pLayer = _pSet->GetLayer(iLayerIndex);
		int iNumCam = pLayer->GetNumberOfCameras();
		int iNumTrig = pLayer->GetNumberOfTriggers();
		for(int iTrig = 0; iTrig < iNumTrig; iTrig++)
		{
			for(int iCam = 0; iCam < iNumCam; iCam++) 
			{
				Image* pImg = pLayer->GetImage(iTrig, iCam);
				int iStride = pImg->PixelRowStride();
				int iRows = pImg->Rows();
				if(iTrig%2==0 && iCam%2==0)
					continue;
				if(iTrig%2==0 && iCam%2==1)
				{
					memset(pImg->GetBuffer()+iStride*iRows, 200, iStride*iRows); // set Cr 200-128
				}
				if(iTrig%2==1 && iCam%2==0)
				{
					memset(pImg->GetBuffer()+iStride*iRows, 50, iStride*iRows); // set Cr 50-128
				}
				if(iTrig%2==1 && iCam%2==1)
				{
					memset(pImg->GetBuffer()+iStride*iRows, 128, iStride*iRows); // set Cr 128-128
				}
			}
		}
	}

	// Create a whole stitched image with patches
	Image* pStitchedImage;
	if(_pSet->IsBayerPattern() && !_pSet->IsSkipDemosaic())
	{
		pStitchedImage = new ColorImage(BGR, false);
	}
	else
	{
		pStitchedImage = new Image();		
	}

	ImgTransform inputTransform;
	inputTransform.Config(dRes, dRes, 0, 0, 0);	
	unsigned int iNumRows = _pSet->GetObjectWidthInPixels();
	unsigned int iNumCols = _pSet->GetObjectLengthInPixels();
	pStitchedImage->Configure(iNumCols, iNumRows, iNumCols, inputTransform, inputTransform, true);
	pStitchedImage->ZeroBuffer();

	for(FeatureListIterator i=_pPanel->beginFeatures(); i!=_pPanel->endFeatures(); i++)
	{
		Box box = i->second->GetBoundingBox();
		_pSet->GetLayer(iLayerIndex)->GetImagePatch(
			pStitchedImage,
			(unsigned int)( (box.p1.y - box.Height()*0.3)/dRes ),
			(unsigned int)( (box.p2.y + box.Height()*0.3)/dRes ),
			(unsigned int)( (box.p1.x - box.Width()*0.3)/dRes ),
			(unsigned int)( (box.p2.x + box.Width()*0.3)/dRes ),
			&setFov);
	}

	pStitchedImage->Save("C:\\Temp\\patchImage.bmp");
	delete pStitchedImage;

	// Create individual patch image only
	int iCount = 0;
	for(FeatureListIterator i=_pPanel->beginFeatures(); i!=_pPanel->endFeatures(); i++)
	{
		Box box = i->second->GetBoundingBox();
		
		// Image patch location and size on the stitched image
		int iStartCol = (int)( (box.p1.y - box.Height()*0.3)/dRes );
		int iStartRow = (int)( (box.p1.x - box.Width()*0.3)/dRes );
		int iCols = (int)( box.Height()*1.6/dRes );
		int iRows = (int)( box.Width()*1.6/dRes );

		Image* pImg;
		int iBytePerPIxel = 1;
		if(_pSet->IsBayerPattern() && !_pSet->IsSkipDemosaic())
		{
			pImg = new ColorImage(BGR, false);
		}
		else
		{
			pImg = new Image();
		}

		// The image that will hold the image patch 
		// Its transform need not match patch information
		ImgTransform trans;
		pImg->Configure(iCols, iRows, iCols, trans, trans, true);
		_pSet->GetLayer(iLayerIndex)->GetImagePatch(
			pImg->GetBuffer(), iCols, iStartRow, iStartCol, iRows, iCols,
			&setFov);

		string s;
		char cTemp[100];
		sprintf_s(cTemp, 100, "C:\\Temp\\Temp\\ID%d.bmp", iCount);
		s.append(cTemp);
		pImg->Save(s);
		delete pImg;

		iCount++;
	}
}

void PanelAligner::TestSingleImagePatch()
{
	// Layer index
	int iLayerIndex = 2;

	// Image patch location and size on the stitched image
	int iStartCol = 625;
	int iStartRow = 2391;
	int iCols = 51;
	int iRows = 101;

	MosaicDM::FOVPreferSelected setFov;
	setFov.preferTB = TOPFOV;
	setFov.preferLR = RIGHTFOV;

	Image* pImg;
	int iBytePerPIxel = 1;
	if(_pSet->IsBayerPattern() && !_pSet->IsSkipDemosaic())
	{
		pImg = new ColorImage(BGR, false);
	}
	else
	{
		pImg = new Image();
	}

	// The image that will hold the image patch 
	// Its transform need not match patch information
	ImgTransform trans;
	pImg->Configure(iCols, iRows, iCols, trans, trans, true);
	_pSet->GetLayer(iLayerIndex)->GetImagePatch(
		pImg->GetBuffer(), iCols, iStartRow, iStartCol, iRows, iCols,
		&setFov);

	pImg->Save( "C:\\Temp\\Patch.bmp");

	delete pImg;
}

void PanelAligner::OutputTransforms(string fileName)
{
	ofstream of(fileName.c_str());

	of << std::scientific;

	unsigned int iNumLayer = _pSet->GetNumMosaicLayers();
	for(unsigned int iLayer=0; iLayer<iNumLayer; iLayer++)
	{
		// Get calculated transforms
		MosaicLayer* pLayer = _pSet->GetLayer(iLayer);
		for(unsigned iTrig=0; iTrig<pLayer->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pLayer->GetNumberOfCameras(); iCam++)
			{
				of << "L" << iLayer 
				<< "_T" << iTrig 
				<< "_C" << iCam
				<< ",";

				ImgTransform t = pLayer->GetImage(iTrig, iCam)->GetTransform();

				for(int j=0; j<8; j++)
				{
					if(j!=0) of << ",";
					of << t.GetItem(j);
				}
				of <<  std::endl;
			}
		}
	}

	of.close();
}

void PanelAligner::OutputFiducialForSolver(string fileName)
{
	ofstream of(fileName.c_str());
	
	of << "Fiducial ID," 
		<< "Cad X(mm),"
		<< "Cad Y(mm),"
		<< "Layer,"
		<< "Trigger,"
		<< "Camera,"
		<< "Correlation Score,"
		<< "Ambiguous,"
		<< "Column Offset (pixel),"
		<< "Row Offset (pixel)"
		<< std::endl;

	// Get the fiducial information
	PanelFiducialResultsSet* pFidResultsSet = GetFidResultsSetPoint();
	for(int i=0; i<pFidResultsSet->Size(); i++)	// For each panel fiducial 
	{
		// Pick one alignment
		PanelFiducialResults* results = pFidResultsSet ->GetPanelFiducialResultsPtr(i);
		list<FidFovOverlap*>* resultList = results->GetFidOverlapListPtr();
		for(list<FidFovOverlap*>::iterator j = resultList->begin(); j != resultList->end(); j++)
		{
			if((*j)->IsProcessed() && (*j)->IsGoodForSolver() && (*j)->GetWeightForSolver()>0)
			{
				CorrelationResult result = (*j)->GetCoarsePair()->GetCorrelationResult();
				of << (*j)->GetFiducialIndex() << ","
					<< (*j)->GetFiducialXPos()*1000 << ","
					<< (*j)->GetFiducialYPos()*1000 << ","
					<< (*j)->GetMosaicLayer()->Index() << "," 
					<< (*j)->GetTriggerIndex() << ","
					<< (*j)->GetCameraIndex() << ","
					<< result.CorrCoeff << ","
					<< result.AmbigScore << ","
					<< result.CorrCoeff << ","
					<< result.RowOffset 
					<< std::endl;
			}
		}
	}

	of.close();
}

#pragma endregion


#pragma region FiducailResultCheck class

///////////////////////////////////////////////////////
//	FiducialResultCheck Class
// Check the validation of fiducial alignment results
///////////////////////////////////////////////////////
FiducialResultCheck::FiducialResultCheck(PanelFiducialResultsSet* pFidSet, RobustSolver* pSolver)
{
	_pFidSet = pFidSet;
	_pSolver = pSolver;
}

// Check the alignment of fiducial, mark out any outlier, 
// Only works well when outliers are minority
// return	 1	: success
//			-1	: marked out Outliers only
//			-2	: not marked out exceptions only
//			-3	: Both Oulier and exception
//			-4	: All fiducial distance are out of scale range


int FiducialResultCheck::CheckFiducialResults()
{
	list<FiducialDistance> fidDisList;

	// Calculate all valid distances of all alignment pairs for different physical fiducials 
	int iNumPhyFid = _pFidSet->Size();
	for(int i=0; i<iNumPhyFid; i++)
	{
		for(int j=i+1; j<iNumPhyFid; j++) // j should be bigger than i
		{
			// Alignment results for two different physical fiducials 
			list<FidFovOverlap*>* pResults1 = _pFidSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
			list<FidFovOverlap*>* pResults2 = _pFidSet->GetPanelFiducialResultsPtr(j)->GetFidOverlapListPtr();

			// Calculate distance of two alignments for different physical fiducial based on transforms
			for(list<FidFovOverlap*>::iterator m = pResults1->begin(); m != pResults1->end(); m++)
			{
				for(list<FidFovOverlap*>::iterator n = pResults2->begin(); n != pResults2->end(); n++)
				{
					ImgTransform trans1 = _pSolver->GetResultTransform(
						(*m)->GetMosaicLayer()->Index(), (*m)->GetTriggerIndex(), (*m)->GetCameraIndex());
					ImgTransform trans2 = _pSolver->GetResultTransform(
						(*n)->GetMosaicLayer()->Index(), (*n)->GetTriggerIndex(), (*n)->GetCameraIndex());
					FiducialDistance fidDis(*m, trans1, *n, trans2);
					if(fidDis._bValid)
						fidDisList.push_back(fidDis);
				}
			}
		}
	}

	// Calculate normal scale = stitched panle image/CAD image
	// When only 3 distances (3 fiducials with 1 alignments each) 
	// or 4 distances (2 fiducials with 2 alignments each) available
	// one alignement outlier will lead to 2 wrong distnsce
	// Therefore, it makes the normal scale go wrong
	bool bNormlized = false;		
	int iCount = 0;
	if(fidDisList.size() > 4)
	{
		// Calcualte mean and variance
		double dSum = 0;
		double dSumSquare = 0;
		for(list<FiducialDistance>::iterator i = fidDisList.begin(); i != fidDisList.end(); i++)
		{
			if(i->_bValid)
			{
				double dScale = i->CalTranScale();
				if(fabs(dScale-1) < CorrelationParametersInst.dMaxPanelCadScaleDiff) // Ignore outliers
				{
					dSum += dScale;
					dSumSquare += dScale*dScale;
					iCount++;
				}
			}
		}

		if(fidDisList.size()-iCount > 0)
			LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): %d out of %d fiducial distance(s) on panel/CAD Scale is out of range", 
				fidDisList.size()-iCount, fidDisList.size()); 

		// Calculate normal scale
		double dNormScale = 1;
		if(iCount==0)					// Failed
			return(-4);
		else if(iCount==1 || iCount==2)	// 1 or 2 values available
			 dNormScale = dSum/iCount;
		else							// 3 or more values available
		{	// refine
			dSum /= iCount;
			dSumSquare /= iCount;
			double dVar = sqrt(dSumSquare - dSum*dSum);
			iCount = 0;
			double dSum2 = 0;
			for(list<FiducialDistance>::iterator i = fidDisList.begin(); i != fidDisList.end(); i++)
			{
				if(i->_bValid)
				{
					double dScale = i->CalTranScale();
					if(fabs(dScale-dSum) <= dVar)
					{
						dSum2 += dScale;
						iCount++;
					}
				}
			}
			dNormScale = dSum2/iCount;
		}

		// Adjust distance base on transform
		for(list<FiducialDistance>::iterator i = fidDisList.begin(); i != fidDisList.end(); i++)
		{
			if(i->_bValid)
			{
				i->NormalizeTransDis(dNormScale);
			}
		}

		bNormlized = true;
	}

	// Mark alignment outlier out based on distance/scale check
	double dMaxScale = CorrelationParametersInst.dMaxFidDisScaleDiff;
	if(!bNormlized) 
		dMaxScale += CorrelationParametersInst.dMaxPanelCadScaleDiff;

	int iOutlierCount = 0;
	for(int i=0; i<iNumPhyFid; i++) // for each physical fiducial
	{
		list<FidFovOverlap*>* pResults = _pFidSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++) // For each alignment
		{
			// Outlier check
			int iCount1=0, iCount2=0;
			for(list<FiducialDistance>::iterator m = fidDisList.begin(); m != fidDisList.end(); m++)
			{
				if(m->_bValid && !m->_bFromOutlier && m->IsWithOverlap(*j))
				{
					iCount1++;
					double dScale = m->CalTranScale();
					if(fabs(1-dScale) > dMaxScale)
					{ 
						iCount2++;
					}
				}	
			}

			// If it is an alignment outlier
			if(iCount2 >=2 && (double)iCount2/(double)iCount1>0.5)
			{
				LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): FidOverlap (Layer=%d, Trig=%d, Cam=%d) is outlier base on consistent check, %d out %d scale are out of rang", 
					(*j)->GetMosaicLayer()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex(),
					iCount2, iCount1);

				// Mark all distances related to the aligmment outlier out
				for(list<FiducialDistance>::iterator m = fidDisList.begin(); m != fidDisList.end(); m++)
				{
					if(m->_bValid && !m->_bFromOutlier &&m->IsWithOverlap(*j))
					{
						m->_bFromOutlier = true;
					}
				}

				// The alignment outlier should not be used for solver
				(*j)->SetIsGoodForSolver(false);

				iOutlierCount++;
			}
		}
	}

	// Exception that is not marked as outlier
	int iExceptCount = 0;
	for(list<FiducialDistance>::iterator m = fidDisList.begin(); m != fidDisList.end(); m++)
	{
		if(m->_bValid && !m->_bFromOutlier && fabs(1-m->CalTranScale())>dMaxScale)
			iExceptCount++;
	}
	if(iExceptCount>0)
		LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): There are %d exception distances not from marked outlier(s)", iExceptCount);

	// Check consistent of alignments for each physical fiducial
	// Assumption: at most one outlier exists for each physical fiducial
	for(int i=0; i<iNumPhyFid; i++)
	{
		list<FidFovOverlap*>* pResults = _pFidSet->GetPanelFiducialResultsPtr(i)->GetFidOverlapListPtr();
		if(pResults->size() == 1) // No consistent check can be done
			continue;

		iCount = 0;
		double dSumX=0, dSumY=0;
		double dSumXSq=0, dSumYSq=0;
		for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
		{
			if((*j)->IsProcessed() && (*j)->IsGoodForSolver() && (*j)->GetWeightForSolver()>0)
			{
				// Calcualte the fiducail location based on alignment
				ImgTransform trans = _pSolver->GetResultTransform(
					(*j)->GetMosaicLayer()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex());
				double x, y;
				(*j)->CalFidCenterBasedOnTransform(trans, &x, &y);

				dSumX += x;
				dSumY += y;
				dSumXSq += x*x;
				dSumYSq += y*y;
				iCount++;
			}
		}
		if(iCount<=1) // No consistent check can be done
			continue;

		dSumX /= iCount;
		dSumY /= iCount;
		dSumXSq /= iCount;
		dSumYSq /= iCount;
		double dVarX= sqrt(dSumXSq-dSumX*dSumX);
		double dVarY= sqrt(dSumYSq-dSumY*dSumY);
		
		// If there is only one outlier and all other alignments are on the same physical location 
		// The outlier's physical position away from all other alignment = dAdjustScale*variance
		double dAdjustScale = (double)iCount/sqrt((double)iCount-1);
		double dAdjustDisX = dVarX*dAdjustScale;
		double dAdjustDisY = dVarY*dAdjustScale;

		// If count==2, only exceptions can be checked, no outlier can be identified
		if(iCount==2)
		{
			if(dAdjustDisX > CorrelationParametersInst.dMaxSameFidInConsist ||
				dAdjustDisY > CorrelationParametersInst.dMaxSameFidInConsist)
			{
				iExceptCount++;
				LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): Two alignments for fiducial #%d are inconsistent ", i);
			}
		}

		// Count >=3, mark outlier out if there is some
		if(iCount>=3)
		{
			// If there is outlier 
			if(dAdjustDisX > CorrelationParametersInst.dMaxSameFidInConsist ||
				dAdjustDisY > CorrelationParametersInst.dMaxSameFidInConsist)
			{
				for(list<FidFovOverlap*>::iterator j = pResults->begin(); j != pResults->end(); j++)
				{
					if((*j)->IsProcessed() && (*j)->IsGoodForSolver() && (*j)->GetWeightForSolver()>0)
					{
						// Calcualte the fiducail location based on alignment
						ImgTransform trans = _pSolver->GetResultTransform(
							(*j)->GetMosaicLayer()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex());
						double x, y;
						(*j)->CalFidCenterBasedOnTransform(trans, &x, &y);

						// If it is outlier based on consistent check
						if((dAdjustDisX>CorrelationParametersInst.dMaxSameFidInConsist && fabs(x-dSumX)>dVarX) || 
							(dAdjustDisY>CorrelationParametersInst.dMaxSameFidInConsist && fabs(y-dSumY)>dVarY))
						{
							(*j)->SetIsGoodForSolver(false);
							iOutlierCount++;
							LOG.FireLogEntry(LogTypeDiagnostic, "FiducialResultCheck::CheckFiducialResults(): FidOverlap (Layer=%d, Trig=%d, Cam=%d) is outlier base on consistent check", 
								(*j)->GetMosaicLayer()->Index(), (*j)->GetTriggerIndex(), (*j)->GetCameraIndex()); 
						}
					}
				}
			}
		}
	}

	if(iOutlierCount>0 && iExceptCount==0)
		return(-1);
	else if(iOutlierCount==0 && iExceptCount>0)
		return(-2);
	else if(iOutlierCount>0 && iExceptCount>0)
		return(-3);
	else
		return(1);
}

#pragma endregion

#pragma region QX functions
bool PanelAligner::CreateQXPanel(double dPanelSizeX, double dPanelSizeY, double dPixelSize)
{
	_bOwnMosaicSetPanel = true;
	if(_pPanel != NULL) 
		delete _pPanel;
	_pPanel = new Panel(dPanelSizeX, dPanelSizeY, dPixelSize, dPixelSize);

	return(true);
}

// Create mosaicset, must after CreateQXPanel()
bool PanelAligner::CreateQXMosaicSet(
	double* pdTrans, double *pdTrigs, 
	unsigned int iNumTrigs, unsigned int iNumCams,
	double dOffsetX, double dOffsetY,
	unsigned int iTileCols, unsigned int iTileRows,
	int iBayerType)
{
	// Validation check
	if(_pPanel == NULL)
		return(false);

	// Delete the exist one 
	if(_pSet != NULL)
		delete _pSet;

	// Create mosaicset
	bool bSkipDemosaic = true;
	bool bBayerPattern = true;
	bool bOwnBuffers = false;
	_pSet = new MosaicSet(
		_pPanel->xLength(), _pPanel->yLength(), 
        iTileCols, iTileRows, iTileCols, 
		_pPanel->GetPixelSizeX(), _pPanel->GetPixelSizeY(), 
        bOwnBuffers,
        bBayerPattern, iBayerType, bSkipDemosaic);


	// Add a mosaic layer
    bool bFiducialAllowNegativeMatch = false; // Bright field not allow negavie match
    bool bAlignWithCAD = false;
    bool bAlignWithFiducial = true;
    bool bFiducialBrighterThanBackground = true;
    unsigned int deviceIndex = 0;
	MosaicLayer* pLayer = _pSet->AddLayer(iNumCams, iNumTrigs, bAlignWithCAD, bAlignWithFiducial, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch, deviceIndex);

	// Set subDevice
    if (iNumCams > 8)
    {
        list<unsigned int> iLastCams;
		iLastCams.push_back(7); 
		iLastCams.push_back(15);
        _pSet->AddSubDeviceInfo(deviceIndex, iLastCams);
    }

	// QX (U, v)->(x,y) to Cyberstitch (U, v)->(x,y) conversion matrix
	double dPanelHeight = _pPanel->xLength();
	unsigned int iImageRows = _pSet->GetImageHeightInPixels();
    double leftM[8] = { 
        0, -1, dPanelHeight-dOffsetY, 
        1, 0, dOffsetX,
        0, 0};

    double rightM[8] = {
        0, 1, 0,
        -1, 0, iImageRows-1,
        0, 0};

    double tempM[8];
    double camM[8];
    double fovM[9];	// Must be 9

    for (unsigned int iCam = 0; iCam < iNumCams; iCam++) // For each camera
    {
        // Calculate camera transform for first trigger
        MultiProjective2D(leftM, pdTrans+iCam*8, tempM);
        MultiProjective2D(tempM, rightM, camM);
		fovM[8] = 1;
        for (int i = 0; i < 8; i++)
            fovM[i] = camM[i];

        for (unsigned int iTrig = 0; iTrig < iNumTrigs; iTrig++) // For each trigger
        {
            // Set transform for each trigger
                fovM[2] -= pdTrigs[iTrig]; // This calcualtion is not very accurate
            
			MosaicTile* pTile = pLayer->GetTile(iTrig, iCam);
            pTile->SetNominalTransform(fovM);

            // For camera model 
            pTile->ResetTransformCamCalibration();
            pTile->ResetTransformCamModel();
            pTile->SetTransformCamCalibrationUMax(_pSet->GetImageWidthInPixels());  // column
            pTile->SetTransformCamCalibrationVMax(_pSet->GetImageHeightInPixels()); // row

            float Sy[16];
            float Sx[16];
            float dSydz[16];
            float dSxdz[16];
            for (unsigned int m = 0; m < 16; m++)
            {
                Sy[m] = 0;
                Sx[m] = 0;
                dSydz[m] = 0;
                dSxdz[m] = 0;
            }
                // S (Nonlinear Parameter for SIM 110 only)
            Sy[3] = (float)-1.78e-5;
            Sy[9] = (float)-1.6e-5;
            Sx[6] = (float)-2.21e-5;
            Sx[12] = (float)-7.1e-6;

                // dS
            double dPupilDistance = 0.3702;
            float fHalfW, fHalfH;
            CalFOVHalfSize(camM, _pSet->GetImageWidthInPixels(), _pSet->GetImageHeightInPixels(), &fHalfW, &fHalfH);
            dSydz[1] = (float)(fHalfW / dPupilDistance);	// dY/dZ
            dSxdz[4] = (float)(fHalfH / dPupilDistance);	// dX/dZ

            pTile->SetTransformCamCalibrationS(0, Sy);
            pTile->SetTransformCamCalibrationS(1, Sx);
            pTile->SetTransformCamCalibrationdSdz(0, dSydz);
            pTile->SetTransformCamCalibrationdSdz(1, dSxdz);
                        
                // Linear part
            pTile->SetCamModelLinearCalib(camM);   
        }
    }

	// Set correlation flag
	CorrelationFlags* pFlag = _pSet->GetCorrelationFlags(0, 0);
	pFlag->SetCameraToCamera(true);
	pFlag->SetTriggerToTrigger(true);

	return(true);
}

// Change production for QX
bool PanelAligner::ChangeQXproduction(
	double dPanelSizeX, double dPanelSizeY, double dPixelSize,
	double* pdTrans, double *pdTrigs, 
	unsigned int iNumTrigs, unsigned int iNumCams,
	double dOffsetX, double dOffsetY,
	unsigned int iTileCols, unsigned int iTileRows,
	int iBayerType)
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Begin panel change over");
	// CleanUp internal stuff for new production
	CleanUp();

	if(!CreateQXPanel(dPanelSizeX, dPanelSizeY, dPixelSize))
		return(false);

	if(!CreateQXMosaicSet(
		pdTrans, pdTrigs, 
		iNumTrigs, iNumCams,
		dOffsetX, dOffsetY,
		iTileCols, iTileRows,
		iBayerType))
		return(false);

	return(ChangeProduction());
}

// For debug
void PanelAligner::SetSeperateProcessStages(bool bValue)
{
	if(_pSet != NULL)
		_pSet->SetSeperateProcessStages(bValue);
}

// Add data to QX image tile
bool PanelAligner::AddQXImageTile(unsigned char* pbBuf, unsigned int iTrig, unsigned int iCam)
{
	return(_pSet->AddRawImage(pbBuf, 0, iCam, iTrig));
}

// Save QX stitched image 
bool PanelAligner::SaveQXStitchedImage(string sStitchedImFile)
{
	_pSet->GetLayer(0)->SaveStitchedImage(sStitchedImFile);

	return(true);
}

// Get transform of QX image tile
bool PanelAligner::GetQXTileTransform(unsigned int iTrig, unsigned int iCam, double dTrans[9])
{
	_pSet->GetLayer(0)->GetTile(iTrig, iCam)->GetTransform(dTrans);

	return(true);
}

void PanelAligner::SaveQXTile(unsigned int iTrig, unsigned int iCam, string sFile)
{
	_pSet->GetLayer(0)->GetTile(iTrig, iCam)->GetImagPtr()->Save(sFile);
}


#pragma region end region