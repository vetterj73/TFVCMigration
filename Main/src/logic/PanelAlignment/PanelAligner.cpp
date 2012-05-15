#include "PanelAligner.h"
#include "OverlapManager.h"
#include "MosaicLayer.h"
#include "MosaicTile.h"
#include "MorphJob.h"
#include "Panel.h"
#include "ColorImage.h"
#include <direct.h> //_mkdir
#include "EquationWeights.h"


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
	_iNumFovProced++;
	
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
	_pMaskSolver = NULL;

	_registeredAlignmentDoneCallback = NULL;
	_pCallbackContext = NULL;

	//_queueMutex = CreateMutex(0, FALSE, "PanelAlignMutex"); // Mutex is not owned
	_queueMutex = CreateMutex(0, FALSE, NULL); // Mutex is not owned

	_iNumFovProced = 0;

	// for debug
	_iPanelCount = 0;
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

	_iNumFovProced = 0;
}

// Change production
bool PanelAligner::ChangeProduction(MosaicSet* pSet, Panel* pPanel)
{
	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():Begin panel change over");
	// CleanUp internal stuff for new production
	CleanUp();

	_pSet = pSet;
	_pSet->RegisterImageAddedCallback(ImageAdded, this) ;

	_pPanel = pPanel;

	_pOverlapManager = new OverlapManager(_pSet, pPanel, CorrelationParametersInst.NumThreads);
		
	// Create solver for all illuminations
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
						_pSet);  // TODO Is it wise to send _pSet to solver??????????????????????????????????????
	}
	else if (bUseCameraModelStitch)
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():State of bUseCameraModelStitch True, %d", bUseCameraModelStitch);
		_pSolver = new RobustSolverCM(	
						&_solverMap, 
						iMaxNumCorrelations,
						_pSet);  // TODO Is it wise to send _pSet to solver??????????????????????????????????????
	}
	else
	{
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction():State of bUseCameraModelStitch False, %d", bUseCameraModelStitch);
		_pSolver = new RobustSolverFOV(	
						&_solverMap, 
						iMaxNumCorrelations, 
						bProjectiveTrans);
	}

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
		iMaxNumCorrelations =  _pOverlapManager->MaxMaskCorrelations();
		// ************************* TODO TODO **************************
		// CHANGE TO RobustSolverCM()   ?!?!?!?!?
		if (bUseCameraModelStitch)
		{
			_pMaskSolver = new RobustSolverCM(	
							&_maskMap, 
							iMaxNumCorrelations,
							_pSet);  
		}
		else
			_pMaskSolver = new RobustSolverFOV(
				&_maskMap, 
				iMaxNumCorrelations, 
				bProjectiveTrans);
	}

	_bMasksCreated = false;
	_bResultsReady = false;

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ChangeProduction(): End panel change over");
	return(true);
}

//Reset for next panel
void PanelAligner::ResetForNextPanel()
{
	_pOverlapManager->ResetforNewPanel();

	_pSolver->Reset();
	// now added just before AddOverlapResultsForIllum()
	//if( CorrelationParametersInst.bUseCameraModelStitch || CorrelationParametersInst.bUseCameraModelIterativeStitch  )
	//{
	//	_pSolver->ConstrainZTerms();
	//	_pSolver->ConstrainPerTrig();
	//}

	if(_pMaskSolver != NULL)
		_pMaskSolver->Reset();

	_bMasksCreated = false;
	_bResultsReady = false;

	_iNumFovProced = 0;

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::ResetForNextPanel()");
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
	// set some useful value.....
	CorrelationParametersInst.bUseCameraModelStitch = bValue;
}
void PanelAligner::UseCameraModelIterativeStitch(bool bValue)
{
	// set some useful value.....
	CorrelationParametersInst.bUseCameraModelIterativeStitch = bValue;
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

#pragma endregion

#pragma region create transforms

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
		AddOverlapResultsForIllum(_pMaskSolver, i, true); // Use fiducials
	}

	// Solve transforms
	_pMaskSolver->SolveXAlgH();
	//if camera model, must flatten fiducials
	_pMaskSolver->FlattenFiducials( GetFidResultsSetPoint() );


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
	// for debug
	_iPanelCount++;

	// Consist check for FovFov alignment of each trigger
	if(CorrelationParametersInst.bFovFovAlignCheck)
	{
		int iCoarseInconsistNum, iFineInconsistNum;
		_pOverlapManager->FovFovAlignConsistCheckForPanel(&iCoarseInconsistNum, &iFineInconsistNum);
	}

	LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms():Begin to create transforms");
	int iNumIllums = _pSet->GetNumMosaicLayers();

	_lastProcessedFids.clear();

	// For debug
	// DisturbFiducialAlignment();

	// Alignment with panel leading edge and without fiducials 
	bool bUseEdgeInfo = false;
	if(CorrelationParametersInst.bDetectPanelEdge)
	{
		// Get panel leading edge information
		double dSlope, dLeftXOffset, dRightXOffset;
		int iLayerIndex4Edge, iTrigIndex, iLeftCamIndex, iRightCamIndex;
		EdgeInfoType type = _pOverlapManager->GetEdgeDetector()->CalLeadingEdgeLocation(
			&dSlope, &dLeftXOffset, &dRightXOffset,
			&iLayerIndex4Edge, &iTrigIndex, 
			&iLeftCamIndex, &iRightCamIndex);

		if(type == INVALID || type == CONFLICTION) // If leading edge detection is failed
		{
			LOG.FireLogEntry(LogTypeError, "PanelAligner::CreateTransforms(): Panel leading edge detection failed with code %d!", (int)type);
		}
		else	// If leading edge detection is success
		{
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Panel leading edge detection success with code %d!", (int)type);
			bUseEdgeInfo = true;

			// Create matrix and vector for solver
			if( CorrelationParametersInst.bUseCameraModelStitch || CorrelationParametersInst.bUseCameraModelIterativeStitch  )
			{
				_pSolver->ConstrainZTerms();
				_pSolver->ConstrainPerTrig();
			}
			for(int i=0; i<iNumIllums; i++)
			{
				// Not use fiducial, not pin panel with calibration 
				// since panel leading edge will be used
				AddOverlapResultsForIllum(_pSolver, i, false, false);
			}

			// Add panel leading edge constraints
			MosaicLayer* pLayer = _pOverlapManager->GetMosaicSet()->GetLayer(iLayerIndex4Edge);
			if(type == LEFTONLYVALID || type == BOTHVALID)
			{
				_pSolver->AddPanelEdgeContraints(pLayer, iLeftCamIndex, iTrigIndex, dLeftXOffset, dSlope);
			}
			if(type == RIGHTONLYVALID || type == BOTHVALID)
			{
				_pSolver->AddPanelEdgeContraints(pLayer, iRightCamIndex, iTrigIndex, dRightXOffset, dSlope);
			}
			// Solve transforms with panel leading edge but without fiducial information
			_pSolver->SolveXAlgH();

			// Get intermediate result transforms
			for(int i=0; i<iNumIllums; i++)
			{
				// Get calculated transforms
				pLayer = _pSet->GetLayer(i);
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
				tempImage2.Configure(iNumCols, iNumRows, iNumCols, 
					trans, trans, true);
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
				sprintf_s(cTemp, 100, "%sStitchedEdgeImage_%d.bmp", CorrelationParametersInst.sDiagnosticPath.c_str(), _iPanelCount);
		
				sFileName.append(cTemp);

				rbg->write(sFileName);

				delete rbg;
			}
		}		
		
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Begin Fiducial search %s !", bUseEdgeInfo ? "with edge":"without edge");
		// Create and Calculate fiducial overlaps for current panel
		_pOverlapManager->DoAlignment4AllFiducial(bUseEdgeInfo);	
		LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): End Fiducial search %s !", bUseEdgeInfo ? "with edge":"without edge");
	}
	
	// After all fiducial overlaps are calculated
	_pOverlapManager->CreateFiducialResultSet(bUseEdgeInfo);

	// If edge information is used but fiducial confidence is very low
	// Fall back to without panel edge information
	if(bUseEdgeInfo)
	{
		double dConfidence = _pOverlapManager->GetFidResultsSetPoint()->CalConfidence();
		if(dConfidence < 0.1)
		{
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): With edge detection, Fiducial condidence is %d!", (int)(dConfidence*100));
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Fall back without edge detection");
			
			// not use edge information
			bUseEdgeInfo = false;
			
			// Do nominal fiducial overlap alignment
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): Begin Fiducial search");
			_pOverlapManager->DoAlignment4AllFiducial(bUseEdgeInfo);
			LOG.FireLogEntry(LogTypeSystem, "PanelAligner::CreateTransforms(): End Fiducial search");

			// After all fiducial overlaps are calculated (It will clear old information automatically)
			_pOverlapManager->CreateFiducialResultSet(bUseEdgeInfo);
		}
	}

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

	// Create matrix and vector for solver
	if( CorrelationParametersInst.bUseCameraModelStitch || CorrelationParametersInst.bUseCameraModelIterativeStitch  )
	{
		_pSolver->ConstrainZTerms();
		_pSolver->ConstrainPerTrig();
	}
	for(int i=0; i<iNumIllums; i++)
	{
		// Use nominal fiducail overlaps if edge info is not available
		AddOverlapResultsForIllum(_pSolver, i, !bUseEdgeInfo);
	}

	// Use current panel fiducial overlaps if edge information is available
	if(bUseEdgeInfo)
		AddCurPanelFidOverlapResults(_pSolver);

	// Solve transforms
	_pSolver->SolveXAlgH();
	//if camera model, must flatten fiducials
	_pSolver->FlattenFiducials( GetFidResultsSetPoint() );

	// TODO populate CM version of gettransfrom
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

	// for debug
	//TestGetImagePatch();
	//TestSingleImagePatch();

	//_pOverlapManager->GetFidResultsSetPoint()->LogResults();
	
	if(CorrelationParametersInst.bSaveTransformVectors)
	{
		_mkdir(CorrelationParametersInst.sDiagnosticPath.c_str());
		char cTemp[255];
		string s;
		sprintf_s(cTemp, 100, "%sTransformVectorX.csv", CorrelationParametersInst.sDiagnosticPath.c_str()); 
		s.clear();
		s.assign(cTemp);
		_pSolver->OutputVectorXCSV(s);
	}
	return(true);
}

// Add overlap results for a certain illumation/mosaic image to solver
void PanelAligner::AddOverlapResultsForIllum(RobustSolver* solver, unsigned int iIllumIndex, bool bUseFiducials, bool bPinPanelWithCalibration)
{
	if(bUseFiducials)
		bPinPanelWithCalibration = false;

	MosaicLayer* pMosaic = _pSet->GetLayer(iIllumIndex);
	for(unsigned iTrig=0; iTrig<pMosaic->GetNumberOfTriggers(); iTrig++)
	{
		for(unsigned iCam=0; iCam<pMosaic->GetNumberOfCameras(); iCam++)
		{
			// Add calibration constraints
			bool bPinFov = false;
			if (bPinPanelWithCalibration &&
				pMosaic->Index() == 0 && iTrig == 1 && iCam == 1)
			{
				bPinFov = true;
			}

			solver->AddCalibationConstraints(pMosaic, iCam, iTrig, bPinFov);

			// Add Fov and Fov overlap results
			FovFovOverlapList* pFovFovList =_pOverlapManager->GetFovFovListForFov(iIllumIndex, iTrig, iCam);
			for(FovFovOverlapListIterator ite = pFovFovList->begin(); ite != pFovFovList->end(); ite++)
			{
				if(ite->IsProcessed() && ite->IsGoodForSolver())
					solver->AddFovFovOvelapResults(&(*ite));
			}

			// Add Cad and Fov overlap results
			CadFovOverlapList* pCadFovList =_pOverlapManager->GetCadFovListForFov(iIllumIndex, iTrig, iCam);
			for(CadFovOverlapListIterator ite = pCadFovList->begin(); ite != pCadFovList->end(); ite++)
			{
				if(ite->IsProcessed() && ite->IsGoodForSolver())
					solver->AddCadFovOvelapResults(&(*ite));
			}

			if(bUseFiducials)
			{
				// Add Fiducial and Fov overlap results
				FidFovOverlapList* pFidFovList =_pOverlapManager->GetFidFovListForFov(iIllumIndex, iTrig, iCam);
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

// Add current panel/(not nominal) fiducial overlap results
void PanelAligner::AddCurPanelFidOverlapResults(RobustSolver* solver)
{
	for(unsigned int k=0; k<_pPanel->NumberOfFiducials(); k++)
	{
		FidFovOverlapList* pFidFovList =_pOverlapManager->GetCurPanelFidFovList4Fid(k);
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
	unsigned int SolverTrigIndex(0);
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
			if( !CorrelationParametersInst.bUseCameraModelStitch  && !CorrelationParametersInst.bUseCameraModelIterativeStitch ) 
				iCount++;
		}
		if( CorrelationParametersInst.bUseCameraModelStitch || CorrelationParametersInst.bUseCameraModelIterativeStitch ) 
			iCount++;
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

// Check fiducial alignment based on SIM calibration
int PanelAligner::FiducialAlignmentCheckOnCalibration()
{
	// Create matrix and vector for solver without fiducial information	
	_pSolver->Reset();
	int iNumIllums = _pSet->GetNumMosaicLayers();
	for(int i=0; i<iNumIllums; i++)
	{
		// Not use fiducial but pin panel with calibration
		AddOverlapResultsForIllum(_pSolver, i, false, true); 
	}

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
	unsigned int iNumIllums = _pSet->GetNumMosaicLayers();
	
	for(unsigned int iIllumIndex=0; iIllumIndex<iNumIllums; iIllumIndex++)
	{
		MosaicLayer* pMosaic = _pSet->GetLayer(iIllumIndex);
		for(unsigned iTrig=0; iTrig<pMosaic->GetNumberOfTriggers(); iTrig++)
		{
			for(unsigned iCam=0; iCam<pMosaic->GetNumberOfCameras(); iCam++)
			{
				// Add Fiducial and Fov overlap results
				FidFovOverlapList* pFidFovList =_pOverlapManager->GetFidFovListForFov(iIllumIndex, iTrig, iCam);
				for(FidFovOverlapListIterator ite = pFidFovList->begin(); ite != pFidFovList->end(); ite++)
				{
					// for debug
					if(iIllumIndex==2 && iTrig==0 && iCam==0)
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
	if(_pSet->IsBayerPattern())
	{
		MosaicLayer* pLayer = _pSet->GetLayer(iLayerIndex);
		int iNumCam = pLayer->GetNumberOfCameras();
		int iNumTrig = pLayer->GetNumberOfTriggers();
		for(int iTrig = 0; iTrig < iNumTrig; iTrig++)
		{
			for(int iCam = 0; iCam < iNumCam; iCam++) 
			{
				Image* pImg = pLayer->GetImage(iCam, iTrig);
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
	if(_pSet->IsBayerPattern())
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
		if(_pSet->IsBayerPattern())
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
	int iLayerIndex = 3;

	// Image patch location and size on the stitched image
	int iStartCol = 3967;
	int iStartRow = 1202;
	int iCols = 2301;
	int iRows = 2301;

	MosaicDM::FOVPreferSelected setFov;


	Image* pImg;
	int iBytePerPIxel = 1;
	if(_pSet->IsBayerPattern())
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