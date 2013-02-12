// QXCaseTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <stdio.h>
#include <string>
#include <list>
#include <iostream>
#include <io.h>
using namespace std;

#include "Bitmap.h"
#include "PanelAligner.h"

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
class QXNominalInfo
{
public:
	unsigned int iNumTrigs;
	unsigned int iNumCams;
	unsigned int iTileColumns;
	unsigned int iTileRows;
	double dOffsetX;
	double dOffsetY;
	double *pdTrigs;
	double *pdTrans;
};
QXNominalInfo _info;
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
	if (_access (_sSimulationFile.c_str(), 0) == -1)
		//||_sSimulationFile.find(".csv'") == string::npos)
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
	if(_pAligner != NULL)
		delete _pAligner;
	CloseHandle(_AlignDoneEvent);

	Output("Done!");
	return 0;
}

#pragma region Set mosaicSet
bool LoadNominalInfo(string file, QXNominalInfo* pInfo)
{
//*** Read data from file 
	char temp[1000];
	ifstream in(file.c_str());

	// Read offset X and Y
	in.getline(temp, 100, ',');
	in.getline(temp, 100, ',');
	pInfo->dOffsetX = -atof(temp);
	in.getline(temp, 100, ',');
	pInfo->dOffsetY = -atof(temp);
	
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
	pInfo->iNumCams = (unsigned int)(transList.size()/8);
	pInfo->pdTrans = new double[transList.size()];
	int iCount = 0;
	for(list<double>::iterator i=transList.begin(); i!= transList.end(); i++)
	{
		pInfo->pdTrans[iCount] = *i;
		iCount++;
	}

	pInfo->iNumTrigs = (unsigned int)trigList.size()+1;
	pInfo->pdTrigs = new double[pInfo->iNumTrigs];
	pInfo->pdTrigs[0] = 0;
	iCount = 1;
	for(list<double>::iterator i=trigList.begin(); i!= trigList.end(); i++)
	{
		pInfo->pdTrigs[iCount] = *i;
		iCount++;
	}

	return(true);
}


#pragma region image load, log and call back
Bitmap* _pBmps = NULL;  
// load all raw image for a mosaic set from a file
bool LoadAllRawImages(string sFolder)
{
	unsigned int iTrigNum = _info.iNumTrigs;
	unsigned int iCamNum = _info.iNumCams;

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
			_pAligner->AddQXImageTile(_pBmps[iTrig*iCamNum+iCam].GetBuffer(), iTrig, iCam);
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
	// Setup aligner
	_pAligner = new PanelAligner();

	// Set up log
	_pAligner->GetLogger()->SetAllLogTypes(true);
	_pAligner->GetLogger()->RegisterLoggingCallback(LoggingCallback);

	// Set up callback
	_pAligner->RegisterAlignmentDoneCallback(OnAlignmentDone, _pAligner);

	// Setup camera model
	_pAligner->UseCameraModelStitch(true);
    _pAligner->UseProjectiveTransform(true);  // projective transform is assumed for camera model stitching
	
	// Skip demosaic
	_pAligner->SetSkipDemosaic(true);

		// For Debug
	//_pAligner->LogTransformVectors(true);
    //_pAligner->LogOverlaps(true);
    //_pAligner->LogFiducialOverlaps(true);

		   // Set up production for aligner
	LoadNominalInfo(_sSimulationFile, &_info);
	if (!_pAligner->ChangeQXproduction(
		_dPanelX, _dPanelY, _dNominalPixelSize,
		_info.pdTrans, _info.pdTrigs, 
		_info.iNumTrigs, _info.iNumCams,
		_info.dOffsetX, _info.dOffsetY,
		_iImCols, _iImRows, _iBayerType))
		return(false);

	// Call after change production
	// Setup threads	
    _pAligner->NumThreads(_iNumThreads);

	// For debug
	//_pAligner->SetSeperateProcessStages(true);
        				
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
		LoadAllRawImages(sFolder);
        
		// Wait alignment to be done
		WaitForSingleObject(_AlignDoneEvent, INFINITE);  
		   
		Output("End stitch cycle");
        iCycleCount++;                   

		// Do the moph and output stitched image
        Output("Begin creating and saving stitched image");			
		string sStitchedImFile;
		sprintf_s(cTemp, 100, "c:\\temp\\Stitched%d.bmp", iCycleCount);
		sStitchedImFile.assign(cTemp);
	
		_pAligner->SaveQXStitchedImage(cTemp);
		Output("End creating and saving stitched image");

		//double dTrans[9];
		//_pAligner->GetQXTileTransform(0, 0,  dTrans);

		ReleaseBmpBufs();  

        // should we do another cycle?
        if (iCycleCount >= _iNumToRun)
			bDone = true;
	}

	return(true);
}