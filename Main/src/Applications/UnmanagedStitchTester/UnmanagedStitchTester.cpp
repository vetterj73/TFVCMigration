// UnmanagedStitchTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <string>
using namespace std;


int main(int argc, char* argv[])
{
	string sPanelFile = "";
	string sSimulationFile = "";

	for(int i=0; i<argc; i++)
	{
		std::string cmd = argv[i];

		if(argv[i] == "-p" && i <= argc-1)
		{
			sPanelFile = argv[i+1];
		}
		else if(argv[i] == "-p" && i <= argc-1)
		{
			sSimulationFile = argv[i+1];
		}
	}


	printf("Done!\n");
	return 0;
}

