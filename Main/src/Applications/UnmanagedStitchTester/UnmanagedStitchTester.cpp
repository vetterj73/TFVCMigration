// UnmanagedStitchTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <string>
using namespace std;

#include "XmlUtils.h"
#include "Panel.h"
#include "SIMAPI.h"

Feature* CreateFeatureFromNode(XmlNode pNode);
Panel* LoadPanelDescription(string sPanelFile);
bool bInitialCoreApi(bool bSimulated, string sSimulationFile);

void main(int argc, char* argv[])	
{
	string _sPanelFile = "";
	string _sSimulationFile = "";
	Panel* _pPanel = NULL;
	bool _bSimulated = false;

	// Paramter input
	for(int i=0; i<argc; i++)
	{
		string cmd = argv[i];

		if(cmd == "-p" && i <= argc-1)
		{
			_sPanelFile = argv[i+1];
		}
		else if(cmd == "-s" && i <= argc-1)
		{
			_sSimulationFile = argv[i+1];
			_bSimulated = true;
		}
	}

	// Only support simulation mode
	if(!_bSimulated)
	{
		printf("No simulation file is available!");
		return;
	}

	// Load panel description
	CoInitialize(NULL);
	_pPanel = LoadPanelDescription(_sPanelFile);
	CoUninitialize();
	if(_pPanel == NULL)
	{
		printf("Failed to load panel file!");
		return;
	}

	// Initial coreApi
	if(!bInitialCoreApi(_bSimulated, _sSimulationFile))
	{
		printf("Failed to initial coreApi");
		return;
	}

	// Clean up
	if(_pPanel != NULL)
		delete _pPanel;

	SIMAPI::SIMCore::RemoveCoreAPI();

	printf("Done!\n");
}

// Load panel description file
Panel* LoadPanelDescription(string sPanelFile)
{
	Panel* pPanel = NULL;
	// Load panel Xml file
	using namespace MSXML2;
	IXMLDOMDocument2Ptr pDOM;
	HRESULT hr = pDOM.CreateInstance(__uuidof(DOMDocument60));
	if(FAILED(hr))
	{
		printf("Could not instantiate XML DOM!");
		return NULL;
	}

	if(pDOM->load(sPanelFile.c_str()) == VARIANT_FALSE)
	{
		printf("Could not load panel file: \'%s\'", sPanelFile.c_str() );
		return NULL;
	}
	
	try{
		// Create panel object
		string s = pDOM->selectSingleNode("//PanelSize/Width")->Gettext();
		double dLengthX = atof(s.c_str());

		s = pDOM->selectSingleNode("//PanelSize/Height")->Gettext();
		double dLengthY = atof(s.c_str());

		pPanel = new Panel(dLengthX, dLengthY, 1.7e-5, 1.7e-5);

		// Add features
		XmlNode pNode = pDOM->selectSingleNode("//Features");
		XmlNodeList pNodeList = pNode->selectNodes("anyType");
		for(int i=0; i<pNodeList->length; i++)
		{
			Feature* pFeature = CreateFeatureFromNode(pNodeList->item[i]);
			pPanel->AddFeature(pFeature);
		}

		// Add fiducials 
		pNode = pDOM->selectSingleNode("//Fiducials");
		pNodeList = pNode->selectNodes("anyType");
		for(int i=0; i<pNodeList->length; i++)
		{
			Feature* pFeature = CreateFeatureFromNode(pNodeList->item[i]);
			pPanel->AddFiducial(pFeature);
		}
	}
	catch(...)
	{
		printf("Faled to read panel filee: \'%s\'", sPanelFile.c_str());
	}

	pDOM.Release();
	return(pPanel);
}

// Create a fearue point from
Feature* CreateFeatureFromNode(XmlNode pNode)
{
	Feature *pFeature = NULL;			
		
	try
	{
		// Common items for all types of fearures
		string s =pNode->selectSingleNode("PositionX")->Gettext();
		double dPosX = atof(s.c_str());

		s =pNode->selectSingleNode("PositionY")->Gettext();
		double dPosY = atof(s.c_str());			
		
		s =pNode->selectSingleNode("ReferenceID")->Gettext();
		unsigned int iID = atoi(s.c_str());
		
		// For different type (not completed);
		string type = XmlUtils::GetStringAttr("xsi:type", pNode);
		if(type == "CSIMRectangle")
		{	// Rectangle feature	
			s =pNode->selectSingleNode("Rotation")->Gettext();
			double dAngle = atof(s.c_str());

			s =pNode->selectSingleNode("SizeX")->Gettext();
			double dSizeX = atof(s.c_str());

			s =pNode->selectSingleNode("SizeY")->Gettext();
			double dSizeY = atof(s.c_str());

			s =pNode->selectSingleNode("SizeZ")->Gettext();
			double dSizeZ = atof(s.c_str());

			pFeature = new RectangularFeature(iID, dPosX, dPosY, dAngle, dSizeX, dSizeY, dSizeZ);
		}
		else if(type == "CSIMDisc")
		{	// Disc feature
			s =pNode->selectSingleNode("Diameter")->Gettext();
			double dDiameter = atof(s.c_str());

			pFeature = new DiscFeature(iID, dPosX, dPosY, dDiameter);
		}
		else
		{
			// not support type
		}
	}
	catch(...)
	{
		printf("Node to feature conversion failed!");
	}

	return(pFeature);
}

// Frame done callback
void OnFrameDone(int device, SIMSTATUS status, class SIMAPI::CSIMFrame* frame, void* context)
{
	printf("Image Ready\n");
}

// Device acquistion doene callback
void OnAcquisitionDone(int device, SIMSTATUS status, int Count, void* context)
{
}

// Initial coreApi
bool bInitialCoreApi(bool bSimulated, string sSimulationFile)
{
	// Validation check
	SIMSTATUS status = SIMAPI::SIMCore::InitializeCoreAPI(bSimulated, sSimulationFile.c_str());
	if(status != SIMSTATUS_SUCCESS)
	{
		printf("Init Failed with %d\n", status);
		return false;
	}
	if(SIMAPI::SIMCore::NumberOfDevices() < 1)
	{
		printf("No SIM Devices Available with %d.\n", status);
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
			printf("Could not %d allocate buffers!\n", iBufCount);
			return false;
		}

		// Call back function
		pDevice->RegisterAcquisitionCallback(OnFrameDone, pDevice);
		pDevice->RegisterAcquisitionDoneCallback(OnAcquisitionDone, pDevice);
	}

	return true;
}

