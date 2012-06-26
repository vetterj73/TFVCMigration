#pragma once

#include <list>
using std::list;


#include "opencv\cxcore.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "EdgeDetectStructDef.h"

bool FindLeadingEdge(IplImage* pImage, StPanelEdgeInImage* ptParam);

bool RobustPixelLineFit(
	const list<int>* pSetX, const list<int>* pSetY, 
	int iMaxIterations, double dMaxMeanAbsRes, 
	double* pdSlope, double* pdOffset);

bool PixelLineFit(
	const list<int>* pSetX, const list<int>* pSetY, 
	double* pdSlope, double* pdOffset);

// For debug
void LogMessage(char* pMessage);
