// UnmanagedStitchTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <string>
#include <fstream>
#include <iostream>
#include <time.h>
using namespace std;

#include "XmlUtils.h"
#include "Panel.h"
#include "SIMAPI.h"
#include "MosaicSet.h"
#include "PanelAligner.h"
#include "ConfigMosaicSet.h"

// "using namespace SIMAPI" will lead confustion of Panel definition (CoreAPI has a panel class)

bool LoadPanelDescription(string sPanelFile);
bool bInitialCoreApi(bool bSimulated, string sSimulationFile);
void SetupMosaic(Panel* pPanel, bool bOwnBuffers, bool bMaskForDiffDevices);
bool SetupAligner();
void Output(const char* message);
bool RunStitch();

Panel* _pPanel = NULL;
MosaicSet* _pMosaicSet = NULL;
PanelAligner* _pAligner = NULL;

unsigned int _iInputImageColumns = 2592;
unsigned int _iInputImageRows = 1944;

bool _bBayerPattern = false;
int _iBayerType = 1; // GBRG, 
bool _bSkipDemosaic = true;

bool _bContinuous = false;
bool _bOwnBuffers = true; /// Must be true because we are release buffers immediately.
bool _bMaskForDiffDevices = false;
bool _bAdjustForHeight = true;
bool _bUseProjective = true;
bool _bUseCameraModel = false;
bool _bUseIterativeCameraModel = false;
bool _bSeperateProcessStages = false;
bool _bUseTwoPassStitch = false;
unsigned int _iNumThreads = 8;
bool _bDetectPanelEdge = false;	
int _iNumToRun = 1;
int _iLayerIndex4Edge = 0;	

bool _bSimulated = false;
unsigned int _iLayerIndex1 = 0;
unsigned int _iLayerIndex2 = 0;
int _iNumAcqsComplete = 0;

HANDLE _AlignDoneEvent;
clock_t sTime;

void main(int argc, char* argv[])	
{
	string _sPanelFile = "";
	string _sSimulationFile = "";	

	remove("C:\\Temp\\UMCybetstitch.log");
	sTime = clock();

	_AlignDoneEvent = CreateEvent(NULL, FALSE, FALSE, NULL); // Auto Reset, nonsignaled

	// Paramter input
	for(int i=0; i<argc; i++)
	{
		string cmd = argv[i];

		if(cmd == "-p" && i <= argc-1)
			_sPanelFile = argv[i+1];
		else if(cmd == "-s" && i <= argc-1)
		{
			_sSimulationFile = argv[i+1];
			_bSimulated = true;
		}
		else if (cmd == "-c")
            _bContinuous = true;
        else if (cmd == "-m")
			_bMaskForDiffDevices = true;
        else if (cmd == "-bayer")
			_bBayerPattern = true;
        else if (cmd == "-nw")
			_bUseProjective = false;
        else if (cmd == "-nh")
			_bAdjustForHeight = false;
		else if (cmd == "-cammod")
			_bUseCameraModel = true;
		else if (cmd == "-de")
			_bDetectPanelEdge = true;
        else if (cmd == "-iter")
			_bUseIterativeCameraModel = true;
        else if (cmd == "-sps")
			_bSeperateProcessStages = true;
        else if (cmd == "-twopass")
            _bUseTwoPassStitch = true;
        else if (cmd == "-skipD")
			_bSkipDemosaic = true;
		else if (cmd == "-t" && i <= argc-1)
			_iNumThreads = atoi(argv[i + 1]);        
		else if (cmd == "-n" && i <= argc-1)
			_iNumToRun = atoi(argv[i + 1]);
	}

	// Only support simulation mode
	if(!_bSimulated)
	{
		Output("No simulation file is available!");
		return;
	}

	// Load panel description
	CoInitialize(NULL);
	bool bFlag = LoadPanelDescription(_sPanelFile);
	CoUninitialize();
	if(!bFlag)
	{
		Output("Failed to load panel file!");
		return;
	}
	Output("Panel description has loaded");

	// Initial coreApi
	if(!bInitialCoreApi(_bSimulated, _sSimulationFile))
	{
		Output("Failed to initial coreApi");
		return;
	}

	// Set up aligner
	if(!SetupAligner())
		Output("Failed to setup aligner");
	
	// Do Stitch work
	if(!RunStitch())
		Output("Failed to run stitch");

	// Clean up
	if(_pPanel != NULL)
		delete _pPanel;
	if(_pMosaicSet != NULL)
		delete _pMosaicSet;
	if(_pAligner != NULL)
		delete _pAligner;

	SIMAPI::SIMCore::RemoveCoreAPI();

	CloseHandle(_AlignDoneEvent);

	Output("Done!");
}


void LoggingCallback(LOGTYPE LogType, const char* message)
{
	char cTemp[255];
	sprintf_s(cTemp, 255, "%f: %s\n",  (float)(clock() - sTime)/CLOCKS_PER_SEC, message);

	ofstream of("C:\\Temp\\UMCybetstitch.log", ios_base::app); // add
	if(of.is_open())
	{
		of << cTemp;
	}
	of.close();
	
	// This should print to command line...
	printf_s("%s", cTemp);  
}

void Output(const char* message)
{
	LoggingCallback(LogTypeSystem, message);
}

#pragma region panel description

// Create a fearue point from
Feature* CreateFeatureFromNode(XmlNode pNode)
{
	Feature *pFeature = NULL;			
		
	try
	{
		// Common items for all types of fearures
		string s =pNode->selectSingleNode("PositionX")->Gettext();
		double dPosX = atof(s.c_str())/1000;

		s =pNode->selectSingleNode("PositionY")->Gettext();
		double dPosY = atof(s.c_str())/1000;			
		
		s =pNode->selectSingleNode("ReferenceID")->Gettext();
		unsigned int iID = atoi(s.c_str());
		
		// For different type (not completed);
		string type = XmlUtils::GetStringAttr("xsi:type", pNode);
		if(type == "CSIMRectangle")
		{	// Rectangle feature	
			s =pNode->selectSingleNode("Rotation")->Gettext();
			double dAngle = atof(s.c_str());

			s =pNode->selectSingleNode("SizeX")->Gettext();
			double dSizeX = atof(s.c_str())/1000;

			s =pNode->selectSingleNode("SizeY")->Gettext();
			double dSizeY = atof(s.c_str())/1000;

			s =pNode->selectSingleNode("SizeZ")->Gettext();
			double dSizeZ = atof(s.c_str())/1000;

			pFeature = new RectangularFeature(iID, dPosX, dPosY, dAngle, dSizeX, dSizeY, dSizeZ);
		}
		else if(type == "CSIMDisc")
		{	// Disc feature
			s =pNode->selectSingleNode("Diameter")->Gettext();
			double dDiameter = atof(s.c_str())/1000;

			pFeature = new DiscFeature(iID, dPosX, dPosY, dDiameter);
		}
		else
		{
			// not support type
		}
	}
	catch(...)
	{
		Output("Node to feature conversion failed!");
	}

	return(pFeature);
}

// Load panel description file
bool LoadPanelDescription(string sPanelFile)
{
	char cTemp[255];
	// Load panel Xml file
	using namespace MSXML2;
	IXMLDOMDocument2Ptr pDOM;
	HRESULT hr = pDOM.CreateInstance(__uuidof(DOMDocument60));
	if(FAILED(hr))
	{
		Output("Could not instantiate XML DOM!");
		return false;
	}

	if(pDOM->load(sPanelFile.c_str()) == VARIANT_FALSE)
	{
		sprintf_s(cTemp, 255, "Could not load panel file: \'%s\'", sPanelFile.c_str() );
		Output(cTemp);
		return false;
	}
	
	try{
		// Create panel object
		string s = pDOM->selectSingleNode("//PanelSize/Width")->Gettext();
		double dLengthX = atof(s.c_str())/1000;

		s = pDOM->selectSingleNode("//PanelSize/Height")->Gettext();
		double dLengthY = atof(s.c_str())/1000;

		_pPanel = new Panel(dLengthX, dLengthY, 1.7e-5, 1.7e-5);

		// Add features
		XmlNode pNode = pDOM->selectSingleNode("//Features");
		XmlNodeList pNodeList = pNode->selectNodes("anyType");
		for(int i=0; i<pNodeList->length; i++)
		{
			Feature* pFeature = CreateFeatureFromNode(pNodeList->item[i]);
			_pPanel->AddFeature(pFeature);
		}

		// Add fiducials 
		pNode = pDOM->selectSingleNode("//Fiducials");
		pNodeList = pNode->selectNodes("anyType");
		for(int i=0; i<pNodeList->length; i++)
		{
			Feature* pFeature = CreateFeatureFromNode(pNodeList->item[i]);
			_pPanel->AddFiducial(pFeature);
		}
	}
	catch(...)
	{
		sprintf_s(cTemp, 255, "Faled to read panel filee: \'%s\'", sPanelFile.c_str());
		Output(cTemp);
	}

	pDOM.Release();
	return(true);
}



#pragma endregion

#pragma region coreAPI

// Frame done callback
void OnFrameDone(int iDevice, SIMSTATUS status, class SIMAPI::CSIMFrame* pFrame, void* context)
{
	char cTemp[255];
	// Before CoreAPI fixed the bug, device index need to be calculated
	int iDeviceIndex = pFrame->DeviceNumber();
	SIMAPI::ISIMDevice *pDevice = SIMAPI::SIMCore::GetSIMDevice(iDeviceIndex);

	int iTrigIndex = ConfigMosaicSet::TranslateTrigger(pFrame);
	int iCamIndex = pFrame->CameraIndex() - pDevice->FirstCameraEnabled();

    unsigned int iLayer = iDeviceIndex * pDevice->NumberOfCaptureSpecs() +
		pFrame->CaptureSpecNumber();

    _pMosaicSet->AddRawImage(pFrame->Buffer(), iLayer, iCamIndex, iTrigIndex);
	//sprintf_s(cTemp, 255, "Fov (layer=%d, Trig=%d, Cam=%d)", iLayer, iTrigIndex, iCamIndex);
	//Output(cTemp);

    pDevice->ReleaseFrameBuffer(pFrame);
}

// Device acquistion doene callback
void OnAcquisitionDone(int iDeviceINdex, SIMSTATUS status, int Count, void* context)
{
	char cTemp[255];
	sprintf_s(cTemp, 255, "End SIM%d acquisition", iDeviceINdex);
	Output(cTemp);
    _iNumAcqsComplete++;
    // lauch next device in simulation case
    if (_bSimulated && _iNumAcqsComplete < SIMAPI::SIMCore::NumberOfDevices())
    {
		// Thread.Sleep(10000);
		SIMAPI::ISIMDevice *pDevice = SIMAPI::SIMCore::GetSIMDevice(_iNumAcqsComplete);
        sprintf_s(cTemp, 255, "Begin SIM%d acquisition",  _iNumAcqsComplete);
		Output(cTemp);
        if (pDevice->StartAcquisition(SIMAPI::Panel) != 0)
			return;
	}
}

// Initial coreApi
bool bInitialCoreApi(bool bSimulated, string sSimulationFile)
{
	char cTemp[255];
	// Validation check
	SIMSTATUS status = SIMAPI::SIMCore::InitializeCoreAPI(bSimulated, sSimulationFile.c_str());
	if(status != SIMSTATUS_SUCCESS)
	{
		sprintf_s(cTemp, 255, "CoreApi initialization failed with simulation file %s and status=%d", sSimulationFile.c_str(), status);
		Output(cTemp);
		return false;
	}
	if(SIMAPI::SIMCore::NumberOfDevices() < 1)
	{
		sprintf_s(cTemp, 255, "No SIM Devices Available with status=%d.", status);
		Output(cTemp);
		return false;
	}

	// For each device
	int iNumDevice = SIMAPI::SIMCore::NumberOfDevices();
	for(int i=0; i<iNumDevice; i++)
	{
		// Allocate buffers
		SIMAPI::ISIMDevice *pDevice = SIMAPI::SIMCore::GetSIMDevice(i);
		
			// Total trigs
		int iTotalTrigs = 0;
		int iNumCS = pDevice->NumberOfCaptureSpecs();
		for(int j=0; j<iNumCS; j++)
			iTotalTrigs += pDevice->GetSIMCaptureSpec(j)->GetNumberOfTriggers();

			// Allocate buffers
		int iBufCount = 8* iTotalTrigs;
		int iAllocatedBufs = iBufCount;
		pDevice->AllocateFrameBuffers(&iAllocatedBufs);
		if(iAllocatedBufs < iBufCount)
		{
			sprintf_s(cTemp, "Could not %d allocate buffers!", iBufCount);
			Output(cTemp);
			return false;
		}

		// Call back function
		pDevice->RegisterAcquisitionCallback(OnFrameDone, pDevice);
		pDevice->RegisterAcquisitionDoneCallback(OnAcquisitionDone, pDevice);
	}

	return true;
}

#pragma endregion

#pragma region Aligner setup

void SetupMosaic(Panel* pPanel, bool bOwnBuffers, bool bMaskForDiffDevices)
{
	// Create mosaicset
	_pMosaicSet = new MosaicSet(
		pPanel->xLength(), pPanel->yLength(), 
        _iInputImageColumns, _iInputImageRows, _iInputImageColumns, 
		pPanel->GetPixelSizeX(), pPanel->GetPixelSizeY(), 
        bOwnBuffers,
        _bBayerPattern, _iBayerType, _bSkipDemosaic);

	// Set log output
	_pMosaicSet->SetAllLogTypes(true);
	_pMosaicSet->RegisterLoggingCallback(LoggingCallback);

	// Configurate mosaicset
	ConfigMosaicSet::MosaicSetDefaultConfiguration(_pMosaicSet, bMaskForDiffDevices);
}

void OnAlignmentDone(bool status)
{
	SetEvent(_AlignDoneEvent);
}

bool SetupAligner()
{
	// Set up mosaic
	SetupMosaic(_pPanel, _bOwnBuffers, _bMaskForDiffDevices);

	_pAligner = new PanelAligner();
	_pAligner->GetLogger()->SetAllLogTypes(true);
	_pAligner->GetLogger()->RegisterLoggingCallback(LoggingCallback);
    //_pAligner->LogTransformVectors(true);

    // Set up aligner delegate
	_pAligner->RegisterAlignmentDoneCallback(OnAlignmentDone, _pAligner);

    // Set up production for aligner
    _pAligner->NumThreads(_iNumThreads);
    _pMosaicSet->SetThreadNumber(_iNumThreads);
    //_pAligner->LogOverlaps(true);
    //_pAligner->LogFiducialOverlaps(true);
    //_pAligner->LogPanelEdgeDebugImages(true);
    _pAligner->UseProjectiveTransform(_bUseProjective);
	if (_bUseTwoPassStitch)
		_pAligner->SetUseTwoPassStitch(true);
    if (_bUseCameraModel)
    {
		_pAligner->UseCameraModelStitch(true);
        _pAligner->UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
    }
    if (_bUseIterativeCameraModel)
    {
		_pAligner->UseCameraModelIterativeStitch(true);
        _pAligner->UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
	}
    _pMosaicSet->SetSeperateProcessStages(_bSeperateProcessStages);

    // Must after InitializeSimCoreAPI() before ChangeProduction()
    SIMAPI::ISIMDevice *pDevice = SIMAPI::SIMCore::GetSIMDevice(0);
    _pAligner->SetPanelEdgeDetection(_bDetectPanelEdge, _iLayerIndex4Edge, !pDevice->ConveyorRtoL(), !pDevice->FixedRearRail());

    // true: Skip demosaic for Bayer image
    if (_bBayerPattern)
		_pAligner->SetSkipDemosaic(_bSkipDemosaic);
            
	if (!_pAligner->ChangeProduction(_pMosaicSet, _pPanel))
		return(false);

	// Calculate indices
	switch (_pMosaicSet->GetNumMosaicLayers())
    {
	case 1:
		_iLayerIndex1 = 0;
        _iLayerIndex2 = 0;
        break;

	case 2:
		_iLayerIndex1 = 0;
        _iLayerIndex2 = 1;
		break;

    case 4:
		_iLayerIndex1 = 2;
        _iLayerIndex2 = 3;
        break;
    }

    // Set component height if it exist
    double dMaxHeight = 0;
    if (_bAdjustForHeight)
		dMaxHeight = _pPanel->GetMaxComponentHeight();

	if (dMaxHeight > 0)
    {
		bool bSmooth = true;
        unsigned char* heightBuf = _pPanel->GetHeightImageBuffer(bSmooth);
        unsigned int iSpan = _pPanel->GetNumPixelsInY();
        double dHeightRes = _pPanel->GetHeightResolution();
        double dPupilDistance = 0.3702;
        // Need modified based on layers that have component 
        _pMosaicSet->GetLayer(_iLayerIndex1)->SetComponentHeightInfo(heightBuf, iSpan, dHeightRes, dPupilDistance);
        _pMosaicSet->GetLayer(_iLayerIndex2)->SetComponentHeightInfo(heightBuf, iSpan, dHeightRes, dPupilDistance);
    }
        				
	return(true);
}
#pragma endregion

#pragma region run stitch
bool GatherImages()
{
	char cTemp[255];
	if (!_bSimulated)
    {        
		for (int i = 0; i <SIMAPI::SIMCore::NumberOfDevices(); i++)
		{
			SIMAPI::ISIMDevice *pDevice = SIMAPI::SIMCore::GetSIMDevice(i);
			sprintf_s(cTemp, 255, "Begin SIM%d acquisition", i);
			Output(cTemp);
            if (pDevice->StartAcquisition(SIMAPI::Panel) != 0)
				return false;
        }
	}
    else
    {   // launch device one by one in simulation case
        SIMAPI::ISIMDevice *pDevice = SIMAPI::SIMCore::GetSIMDevice(0);
		Output("Begin SIM0 acquisition");
        if (pDevice->StartAcquisition(SIMAPI::Panel) != 0)
			return false;
	}
    return true;
}

bool RunStitch()
{
	char cTemp[255];
	int iCycleCount = 0;
	bool bDone = false;
    while(!bDone)
    {
		_iNumAcqsComplete = 0;
        _pAligner->ResetForNextPanel();
        _pMosaicSet->ClearAllImages();

        Output("Begin stitch cycle...");
        if (!GatherImages())
        {
            Output("Issue with StartAcquisition");
            return false;
        }
        else
        {
			Output("Waiting for Images...");
            WaitForSingleObject(_AlignDoneEvent, INFINITE);
        }

        // Verify that mosaic is filled in...
        if (!_pMosaicSet->HasAllImages())
		{
			Output("The mosaic does not contain all images!");
			return false;
		}
        else
        {
			Output("End stitch cycle");
            iCycleCount++;                   

            Output("Begin morph");
			
			string sStitchedImFile;
			sprintf_s(cTemp, 100, "c:\\temp\\Aftercycle%d.bmp", iCycleCount);
			sStitchedImFile.assign(cTemp);
            _pAligner->Save3ChannelImage(sStitchedImFile,
				_pMosaicSet->GetLayer(_iLayerIndex1)->GetGreyStitchedImage()->GetBuffer(),
				_pMosaicSet->GetLayer(_iLayerIndex2)->GetGreyStitchedImage()->GetBuffer(),
				_pPanel->GetCadBuffer(), //heightBuf,
                _pPanel->GetNumPixelsInY(), _pPanel->GetNumPixelsInX());

            // After a panel is stitched and before aligner is reset for next panel
			PanelFiducialResultsSet* pFidResultSet = _pAligner->GetFidResultsSetPoint();   
		}

        // should we do another cycle?
        if (!_bContinuous && iCycleCount >= _iNumToRun)
			bDone = true;
	}

	return(true);
}
#pragma endregion