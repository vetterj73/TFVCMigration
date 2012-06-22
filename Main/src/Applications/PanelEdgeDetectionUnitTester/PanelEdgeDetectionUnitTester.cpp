// PanelEdgeDetectionUnitTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "EdgeDetectUtil.h"

#define MAX_PATH 200

bool DetectPanelFrontEdge(const char* filePath, const char* fileName)
{
	// Load Image
	IplImage* pImage = cvLoadImage(filePath);

	// Initial parameters
	StPanelEdgeInImage stParam;
	stParam.iLeft = 700;
	stParam.iRight = 2500;
	//stParam.iLeft = 300;
	//stParam.iRight = 1500;
	stParam.iTop = 500;
	stParam.iBottom = 1300;
	stParam.dAngleRange = 2;
	stParam.dMinLineLengthRatio = 0.3;
	stParam.type = BOTTOMEDGE;
	stParam.iDecim = 2;

	// Edge detection
	bool bFlag = FindLeadingEdge(pImage, &stParam);
	char Name[MAX_PATH];
	strncpy_s(Name, "C:\\Temp\\", MAX_PATH);
	strncat_s(Name, fileName, MAX_PATH);
	if(!bFlag)
	{
		cvSaveImage(Name, pImage);
		cvReleaseImage(&pImage);
		return(false);
	}

	//*/ Draw edge on image 
	float dSlope = stParam.dSlope;
	float dStartRow = stParam.dRowOffsetInColumn0;
	CvPoint pt1, pt2;
	pt1.x = 0;
	pt1.y = dStartRow;
	pt2.x = 2500;
	pt2.y = pt1.y +dSlope*pt2.x;
	cvLine( pImage, pt1, pt2, CV_RGB(255,0,0), 1, 8 );
	
	//cvNamedWindow( "Hough", 1 );
    //cvShowImage( "Hough", pImage );

	// Save result image
	cvSaveImage(Name, pImage);
	cvReleaseImage(&pImage);
	//*/

	return(bFlag);
}


int wmain(int argc, char* argv[])
{
	// Test image folder 
	char folder[] = "D:\\JukiSim\\Panel Edege detection\\PanelEdgeTest\\";
	//char folder[] = "D:\\JukiSim\\PanelEdgeSamples\\";
	//char folder[] = "D:\\JukiSim\\TempTest\\";
   	WIN32_FIND_DATA ffd;
	char find[MAX_PATH];
	char filePath[MAX_PATH];
	strncpy_s(find, folder, MAX_PATH);
	strncat_s(find, "*.png", MAX_PATH);

	// Log file
	char cLogBuf[MAX_PATH];

	int iCount = 0;
	int iErrorCount = 0;

	int iStart = ::GetTickCount();

	// Go through all test images
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

		// Front edge detection
		bool bFlag = DetectPanelFrontEdge(filePath, ffd.cFileName);

		iCount++;
		printf("#%d: %s\n", iCount, ffd.cFileName);
		sprintf_s(cLogBuf, "#%d: %s\n", iCount, ffd.cFileName);
		LogMessage(cLogBuf);

		// Log failure case
		if(!bFlag)
		{
			printf("Failed to detect leading edge\n");
			sprintf_s(cLogBuf, "#%d: %s Failed to detect leading edge\n", iCount, ffd.cFileName);
			LogMessage(cLogBuf);
			iErrorCount++;
			//cvWaitKey(0);
		}
		if(FindNextFile(hFind, &ffd)==0)
			bDone = true;
	}
	int iEnd = ::GetTickCount();

	FindClose(hFind);

	// Log result 
	printf("%d out of %d leading edge are missed!\n", iErrorCount, iCount);
	printf("Took %f second\n", (iEnd-iStart)/1000.);
	sprintf_s(cLogBuf, "%d out of %d leading edge are missed!\n", iErrorCount, iCount);
	LogMessage(cLogBuf);
	sprintf_s(cLogBuf, "Took %f second\n", (iEnd-iStart)/1000.);
	LogMessage(cLogBuf);
	printf("Done!");

	return 0;
}

