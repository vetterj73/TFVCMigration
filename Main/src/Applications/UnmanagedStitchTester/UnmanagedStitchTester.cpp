// UnmanagedStitchTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <string>
using namespace std;

#include "XmlUtils.h"
#include "Panel.h"
//#include "SIMAPI.h"

Feature* CreateFeatureFromNode(XmlNode pNode);
bool LoadPanelDescription(string sPanelFile);

Panel* _pPanel = NULL;
int main(int argc, char* argv[])
{
	string _sPanelFile = "";
	string _sSimulationFile = "";
	bool bSimulated = false;

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
			bSimulated = true;
		}
	}

	// Load panel description
	CoInitialize(NULL);
	LoadPanelDescription(_sPanelFile);
	CoUninitialize();

	/*/ Initial coreAPI
	SIMSTATUS status = SIMAPI::SIMCore::InitializeCoreAPI(
	bSimulated, simulationFile.c_str());
	if(status != SIMSTATUS_SUCCESS)
	{
		printf("Init Failed with %d\n", status);
		return 1;
	}
	if(SIMAPI::SIMCore::NumberOfDevices() < 1)
	{
		printf("No SIM Devices Available.\n", status);
		return 1;
	}*/

	// Clean up
	if(_pPanel != NULL)
		delete _pPanel;

	printf("Done!\n");
	return 0;
}

// Load panel description file
bool LoadPanelDescription(string sPanelFile)
{
	// Load panel Xml file
	using namespace MSXML2;
	IXMLDOMDocument2Ptr pDOM;
	HRESULT hr = pDOM.CreateInstance(__uuidof(DOMDocument60));
	if(FAILED(hr))
	{
		printf("Could not instantiate XML DOM!");
		return false;
	}

	if(pDOM->load(sPanelFile.c_str()) == VARIANT_FALSE)
	{
		printf("Could not load panel file: \'%s\'", sPanelFile.c_str() );
		return false;
	}
	
	try{
		// Create panel object
		string s = pDOM->selectSingleNode("//PanelSize/Width")->Gettext();
		double dLengthX = atof(s.c_str());

		s = pDOM->selectSingleNode("//PanelSize/Height")->Gettext();
		double dLengthY = atof(s.c_str());

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
		printf("Faled to read panel filee: \'%s\'", sPanelFile.c_str());
	}

	pDOM.Release();
	return(true);
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