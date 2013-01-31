// QXCaseTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdio.h>
#include <string>
#include <list>
#include <iostream>
#include <io.h>
using namespace std;


#include "Panel.h"
#include "MosaicSet.h"
#include "MosaicTile.h"
#include "MosaicLayer.h"
#include "CorrelationFlags.h"
#include "Bitmap.h"
#include "PanelAligner.h"

using namespace MosaicDM;

void Output(const char* message);
bool SetupAligner();
bool RunStitch();

clock_t _startTime;

// Default paramter for SIM install to front 
int _iBayerType = 1; // GBRG, 
// default parameter for SIM 110
int _iImCols = 2592;
int _iImRows = 1944;
double _dNominalPixelSize = 1.7e-5;

// Required inputs
string _sSimulationFile = "";	
double _dPanelX = 0;
double _dPanelY = 0;

// Optional inputs
int _iNumThreads = 8;
int _iNumToRun = 1;

// Interal variable
Panel* _pPanel = NULL;
MosaicSet* _pMosaicSet = NULL;
PanelAligner* _pAligner = NULL;
HANDLE _AlignDoneEvent;

int main(int argc, char* argv[])
{
	// Parameter input
	for(int i=0; i<argc; i++)
	{
		string cmd = argv[i];

		if(cmd == "-s" && i <= argc-1)
			_sSimulationFile = argv[i+1];
		else if(cmd == "-xs" && i <= argc-1)
			_dPanelX = atof(argv[i+1]);
		else if(cmd == "-ys" && i <= argc-1)
			_dPanelY = atof(argv[i+1]);
		else if(cmd == "-t" && i <= argc-1)
			_iNumThreads = atoi(argv[i+1]);
		else if(cmd == "-n" && i <= argc-1)
			_iNumToRun = atoi(argv[i+1]);
	}	
	
	// Input validation check
	if(_dPanelX==0 || _dPanelY==0)
	{
		Output("Panel Size is not available");
		return(0);
	}
	if (_access (_sSimulationFile.c_str(), 0) == -1 ||
		_sSimulationFile.find(".csv'") == string::npos)
	{
		Output("The simulation file doesn't exist or is invalid");
		return(0);
	}

	// Prepare for log and event
	remove("C:\\Temp\\QXCybetstitch.log");
	_startTime = clock();
	_AlignDoneEvent = CreateEvent(NULL, FALSE, FALSE, NULL); // Auto Reset, nonsignaled
	
	// Set up aligner
	if(!SetupAligner())
	{
		Output("Failed to setup aligner");
		return(0);
	}
	
	// Do Stitch work
	if(!RunStitch())
	{
		Output("Failed to run stitch");
		return(0);
	}

	// Clean up
	if(_pPanel != NULL)
		delete _pPanel;
	if(_pMosaicSet != NULL)
		delete _pMosaicSet;
	if(_pAligner != NULL)
		delete _pAligner;
	CloseHandle(_AlignDoneEvent);

	Output("Done!");
	return 0;
}

#pragma region Set mosaicSet
// leftM[8] * rightM[8] = outM[8]
void MultiProjective2D(double* leftM, double* rightM, double* outM)
{
    outM[0] = leftM[0]*rightM[0]+leftM[1]*rightM[3]+leftM[2]*rightM[6];
	outM[1] = leftM[0]*rightM[1]+leftM[1]*rightM[4]+leftM[2]*rightM[7];
	outM[2] = leftM[0]*rightM[2]+leftM[1]*rightM[5]+leftM[2]*1;
											 
	outM[3] = leftM[3]*rightM[0]+leftM[4]*rightM[3]+leftM[5]*rightM[6];
	outM[4] = leftM[3]*rightM[1]+leftM[4]*rightM[4]+leftM[5]*rightM[7];
	outM[5] = leftM[3]*rightM[2]+leftM[4]*rightM[5]+leftM[5]*1;
											 
	outM[6] = leftM[6]*rightM[0]+leftM[7]*rightM[3]+1*rightM[6];
	outM[7] = leftM[6]*rightM[1]+leftM[7]*rightM[4]+1*rightM[7];
	double dScale = leftM[6]*rightM[2]+leftM[7]*rightM[5]+1*1;

	if(dScale<0.01 && dScale>-0.01)
		dScale = 0.01;

	for(int i=0; i<8; i++)
		outM[i] = outM[i]/dScale;
}

// (Row, Col) -> (x, y)
void Pixel2World(double* trans, double row, double col, double* px, double* py)
{
    double dScale = 1+ trans[6]*row+trans[7]*col;
    dScale = 1/dScale;
    *px = trans[0] * row + trans[1] * col + trans[2];
    *px *= dScale;
    *py = trans[3] * row + trans[4] * col + trans[5];
    *py *= dScale;
}

// Calculate half size of FOV in world space
void CalFOVHalfSize(double* trans, unsigned int iImW, unsigned int iImH, float* pfHalfW, float* pfHalfH)
{
    // calcualte width
    double dLeftx, dLefty, dRightx, dRighty;
    Pixel2World(trans, (iImH - 1) / 2.0, 0, &dLeftx, &dLefty);
    Pixel2World(trans, (iImH - 1) / 2.0, iImW-1, &dRightx, &dRighty);
    *pfHalfW = (float)((dRighty-dLefty)/2.0);

    // calcualte height
    double dTopx, dTopy, dBottomx, dBottomy;
    Pixel2World(trans, 0, (iImW-1)/2.0, &dTopx, &dTopy);
    Pixel2World(trans, iImH - 1, (iImW-1)/2.0, &dBottomx, &dBottomy);
    *pfHalfH = (float)((dBottomx-dTopx)/2.0);
}

bool SetMosaicSetConfig(MosaicSet* pSet, Panel* pPanel, string file)
{
//*** Read data from file 
	char temp[1000];
	ifstream in(file.c_str());

	// Read offset X and Y
	in.getline(temp, 100, ',');
	in.getline(temp, 100, ',');
	double dOffsetX = -atof(temp);
	in.getline(temp, 100, ',');
	double dOffsetY = -atof(temp);
	
	// Skip a line
	for(int i=0; i<8; i++)
		in.getline(temp, 100, ')');

	// Read camera nominal transform
	list<double> transList;
	while(!in.bad())
	{
		in.getline(temp, 100, ',');
		string str;
		str.append(temp);
		if(str.find("Camera") == string::npos)
			break;

		for(int i=0; i<8; i++)
		{
			in.getline(temp, 100, ',');
			double dValue = atof(temp);
			transList.push_back(dValue);
		}
	}

	// read trig list
	list<double> trigList;
	in.getline(temp, 100, ',');
	double dValue = atof(temp);
	trigList.push_back(dValue);
	while(!in.bad())
	{
		in.getline(temp, 100, ',');
		string str;
		str.append(temp);
		if(str.find("Trig") == string::npos)
			break;

		in.getline(temp, 100, ',');
		double dValue = atof(temp);
		trigList.push_back(dValue);
	}
	in.close();

	// Convert into array
	double* pTrans = new double[transList.size()];
	int iCount = 0;
	for(list<double>::iterator i=transList.begin(); i!= transList.end(); i++)
	{
		pTrans[iCount] = *i;
		iCount++;
	}

	double* pTrigs = new double[trigList.size()];
	iCount = 0;
	for(list<double>::iterator i=trigList.begin(); i!= trigList.end(); i++)
	{
		pTrigs[iCount] = *i;
		iCount++;
	}

//*** Setup nominal transforms, s and dsdz
    // Add a mosaic layer
	unsigned int iNumCam = (unsigned int)transList.size()/8;
	unsigned int iNumTrig = (unsigned int)trigList.size() + 1;

    bool bFiducialAllowNegativeMatch = false; // Bright field not allow negavie match
    bool bAlignWithCAD = false;
    bool bAlignWithFiducial = true;
    bool bFiducialBrighterThanBackground = true;
    unsigned int deviceIndex = 0;
	MosaicLayer* pLayer = pSet->AddLayer(iNumCam, iNumTrig, bAlignWithCAD, bAlignWithFiducial, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch, deviceIndex);

	// Set subDevice
    if (iNumCam > 8)
    {
        list<unsigned int> iLastCams;
		iLastCams.push_back(7); 
		iLastCams.push_back(15);
        pSet->AddSubDeviceInfo(deviceIndex, iLastCams);
    }

	// QX (U, v)->(x,y) to Cyberstitch (U, v)->(x,y) conversion matrix
	double dPanelHeight = pPanel->xLength();
	unsigned int iImageRows = pSet->GetImageHeightInPixels();
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

    for (unsigned int iCam = 0; iCam < iNumCam; iCam++) // For each camera
    {
        // Calculate camera transform for first trigger
        MultiProjective2D(leftM, pTrans+iCam*8, tempM);
        MultiProjective2D(tempM, rightM, camM);
		fovM[8] = 1;
        for (int i = 0; i < 8; i++)
            fovM[i] = camM[i];

        for (unsigned int iTrig = 0; iTrig < iNumTrig; iTrig++) // For each trigger
        {
            // Set transform for each trigger
            if (iTrig > 0)
                fovM[2] -= pTrigs[iTrig - 1]; // This calcualtion is not very accurate
            
			MosaicTile* pTile = pLayer->GetTile(iTrig, iCam);
            pTile->SetNominalTransform(fovM);

            // For camera model 
            pTile->ResetTransformCamCalibration();
            pTile->ResetTransformCamModel();
            pTile->SetTransformCamCalibrationUMax(pSet->GetImageWidthInPixels());  // column
            pTile->SetTransformCamCalibrationVMax(pSet->GetImageHeightInPixels()); // row

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
            CalFOVHalfSize(camM, pSet->GetImageWidthInPixels(), pSet->GetImageHeightInPixels(), &fHalfW, &fHalfH);
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

	delete [] pTrans;
	delete [] pTrigs;

	// Set correlation flag
	CorrelationFlags* pFlag = pSet->GetCorrelationFlags(0, 0);
	pFlag->SetCameraToCamera(true);
	pFlag->SetTriggerToTrigger(true);

	return(true);
}


bool SetUpMosaicSet(Panel* pPanel, string sSimulationFile)
{
	bool bSkipDemosaic = true;
	bool bBayerPattern = true;
	bool bOwnBuffers = true;

	// Create mosaicset
	_pMosaicSet = new MosaicSet(
		pPanel->xLength(), pPanel->yLength(), 
        _iImCols, _iImRows, _iImCols, 
		pPanel->GetPixelSizeX(), pPanel->GetPixelSizeY(), 
        bOwnBuffers,
        bBayerPattern, _iBayerType, bSkipDemosaic);

	// Setup mosaicset configuration
	bool bFlag = SetMosaicSetConfig(_pMosaicSet, pPanel, sSimulationFile);

	return(bFlag);
}

#pragma endregion

#pragma region image load, log and call back
Bitmap* _pBmps = NULL;  
// load all raw image for a mosaic set from a file
bool LoadAllRawImages(MosaicSet* pSet, string sFolder)
{
    MosaicLayer* pLayer = pSet->GetLayer(0);
    unsigned int iTrigNum = pLayer->GetNumberOfTriggers();
    unsigned int iCamNum = pLayer->GetNumberOfCameras();

	_pBmps = new Bitmap[iTrigNum*iCamNum];
    for (unsigned int iTrig = 0; iTrig < iTrigNum; iTrig++)
    {
        for (unsigned int iCam = 0; iCam < iCamNum; iCam++)
        {
            string sFile;
			char temp[100];
			sprintf_s(temp, 100, "%s\\Cam%d_Trig%d.bmp", sFolder.c_str(), iCam, iTrig);
			if (_access (temp, 0) == -1) 
				return(false);
			sFile.append(temp);

			_pBmps[iTrig*iCamNum+iCam].read(sFile);
			pSet->AddRawImage(_pBmps[iTrig*iCamNum+iCam].GetBuffer(), 0, iCam, iTrig);
        }
    }

    return (true);
}

// Realse all image buffers
void ReleaseBmpBufs()
{
	if(_pBmps!=NULL)
		delete [] _pBmps;
}

// Log callback
void LoggingCallback(LOGTYPE LogType, const char* message)
{
	char cTemp[255];
	sprintf_s(cTemp, 255, "%f: %s\n",  (float)(clock() - _startTime)/CLOCKS_PER_SEC, message);

	ofstream of("C:\\Temp\\QXCybetstitch.log", ios_base::app); // add
	if(of.is_open())
	{
		of << cTemp;
	}
	of.close();
	
	// This should print to command line...
	printf_s("%s", cTemp);  
}

// log output
void Output(const char* message)
{
	LoggingCallback(LogTypeSystem, message);
}

// Alignment done callback
void OnAlignmentDone(bool status)
{
	SetEvent(_AlignDoneEvent);
}

#pragma endregion

// Setup aligner
bool SetupAligner()
{
	// Set panel
	_pPanel = new Panel(_dPanelX, _dPanelY, _dNominalPixelSize, _dNominalPixelSize);

	// Set up mosaic
	SetUpMosaicSet(_pPanel, _sSimulationFile);

	// Setup aligner
	_pAligner = new PanelAligner();

		// Set up log
	_pAligner->GetLogger()->SetAllLogTypes(true);
	_pAligner->GetLogger()->RegisterLoggingCallback(LoggingCallback);

		// Set up callback
	_pAligner->RegisterAlignmentDoneCallback(OnAlignmentDone, _pAligner);

		// Setup threads	
    _pAligner->NumThreads(_iNumThreads);
    _pMosaicSet->SetThreadNumber(_iNumThreads);

		// Setup camera model
	_pAligner->UseCameraModelStitch(true);
    _pAligner->UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
	
		// Skip demosaic
	_pAligner->SetSkipDemosaic(true);

		// For Debug
	//_pAligner->LogTransformVectors(true);
    //_pAligner->LogOverlaps(true);
    //_pAligner->LogFiducialOverlaps(true);
	_pMosaicSet->SetSeperateProcessStages(true);

		   // Set up production for aligner
	if (!_pAligner->ChangeProduction(_pMosaicSet, _pPanel))
		return(false);
        				
	return(true);
}


bool RunStitch()
{
	char cTemp[255];
	int iCycleCount = 0;
	bool bDone = false;
    while(!bDone)
    {
        _pAligner->ResetForNextPanel();
        _pMosaicSet->ClearAllImages();
		Output("Waiting for Images...");

        // Directly load image from disc
        string sFolder;
		char temp[100];
		int iIndex = (int)_sSimulationFile.find_last_of("\\");
		string sDir =_sSimulationFile.substr(0, iIndex);
		sprintf_s(temp, 100, "%s\\Cycle%d", sDir.c_str(), iCycleCount);
		sFolder.append(temp);
		if((GetFileAttributes(sFolder.c_str())) == INVALID_FILE_ATTRIBUTES)
		{
			Output("Invalid folder");
			break;
		}
		LoadAllRawImages(_pMosaicSet, sFolder);
        
		// Wait alignment to be done
		WaitForSingleObject(_AlignDoneEvent, INFINITE);  
		ReleaseBmpBufs();     

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

			// Do the moph and output stitched image
            Output("Begin creating and saving stitched image");			
			string sStitchedImFile;
			sprintf_s(cTemp, 100, "c:\\temp\\Stitched%d.bmp", iCycleCount);
			sStitchedImFile.assign(cTemp);
	
			_pMosaicSet->GetLayer(0)->SaveStitchedImage(sStitchedImFile);
			Output("End creating and saving stitched image");
		}

        // should we do another cycle?
        if (iCycleCount >= _iNumToRun)
			bDone = true;
	}

	return(true);
}