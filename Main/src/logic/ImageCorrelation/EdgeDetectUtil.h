#pragma once

//#include "Image.h"
#include "opencv\cxcore.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"

#include <list>
using std::list;

enum PanelEdgeType
{
	LEFTEDGE,
	RIGHTEDGE,
	TOPEDGE,
	BOTTOMEDGE
};

// For panel edge detection on a FOV
struct StPanelEdgeInImage
{
	// Input
	int iLeft;					// ROI
	int iRight;
	int iTop;
	int iBottom;
	PanelEdgeType type;			// type of edge
	int iDecim;					// Decim, valid value = 1, 2, 4;
	double dMinLineLengthRatio;	// Ratio of minimum length of edge
	double dAngleRange;			// Angle range of edge
	
	// Output
	int iFlag;					// rResult flag			
	double dStartY;				// Line paramter base on original FOV image
	double dSlope;

	StPanelEdgeInImage()
	{
		iDecim= 2;
		dMinLineLengthRatio = 0.5;
		dAngleRange = 3;
		iFlag = 0;
	}
};

bool FindLeadingEdge(IplImage* pImage, StPanelEdgeInImage* ptParam);


bool RobustPixelLineFit(
	const list<int>* pSetX, const list<int>* pSetY, 
	int iMaxIterations, double dMaxMeanAbsRes, 
	double* pdSlope, double* pdOffset);

bool PixelLineFit(
	const list<int>* pSetX, const list<int>* pSetY, 
	double* pdSlope, double* pdOffset);
