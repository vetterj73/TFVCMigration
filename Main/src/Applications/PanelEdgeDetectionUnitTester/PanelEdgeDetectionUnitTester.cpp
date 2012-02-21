// PanelEdgeDetectionUnitTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "PanelEdgeDetection.h"


#define MAX_PATH 200

bool DetectPanelFrontEdge(const char* filePath, const char* fileName)
{
	// Load Image
	IplImage* pImage = cvLoadImage(filePath);

	// Initial parameters
	StPanelEdgeInImage stParam;
	stParam.iLeft = 700;
	stParam.iRight = 2500;
	stParam.iTop = 500;
	stParam.iBottom = 1300;
	stParam.dAngleRange = 2;
	stParam.dMinLineLengthRatio = 0.3;
	stParam.type = BOTTOMEDGE;

	// Edge detection
	bool bFlag = PanelEdgeDetection::FindLeadingEdge(pImage, &stParam);

	// Dsiaplay edge
	float rho = stParam.dRho;
	float theta = stParam.dTheta;
    CvPoint pt1, pt2;
	double dSlope = tan(theta);
	pt1.x = 0;
    pt1.y = rho;
    pt2.x = 2500;
    pt2.y = pt1.y+ (pt2.x-pt1.x)*dSlope;
	cvLine( pImage, pt1, pt2, CV_RGB(255,0,0), 3, 8 );
	//cvNamedWindow( "Hough", 1 );
    //cvShowImage( "Hough", pImage );

	// Save result image
	char Name[MAX_PATH];
	strncpy_s(Name, "C:\\Temp\\", MAX_PATH);
	strncat_s(Name, fileName, MAX_PATH);
	cvSaveImage(Name, pImage);
	cvReleaseImage(&pImage);

	return(bFlag);
}



int wmain(int argc, char* argv[])
{
	char folder[] = "D:\\JukiSim\\PanelEdgeSamples1\\";
   	WIN32_FIND_DATA ffd;
	char find[MAX_PATH];
	char filePath[MAX_PATH];
	strncpy_s(find, folder, MAX_PATH);
	strncat_s(find, "*.png", MAX_PATH);

	int iCount = 0;

    HANDLE hFind = FindFirstFile(find, &ffd);
    if (INVALID_HANDLE_VALUE == hFind) 
	{
		printf("Could not find any files in the directory!");
		return 1;
	}
	bool bDone = false;
	while(!bDone)
	{
		strncpy_s(filePath, folder, MAX_PATH);
		strncat_s(filePath, ffd.cFileName, MAX_PATH);

		DetectPanelFrontEdge(filePath, ffd.cFileName);

		iCount++;
		printf("#%d: %s\n", iCount, ffd.cFileName);
		//cvWaitKey(0);

		if(FindNextFile(hFind, &ffd)==0)
			bDone = true;
	}
	FindClose(hFind);

	printf("Done!");

	return 0;
}

