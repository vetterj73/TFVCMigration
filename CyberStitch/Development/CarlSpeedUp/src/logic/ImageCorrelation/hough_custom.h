#pragma once


#include "opencv\cv.h"
#include "opencv\cxcore.h"

CV_IMPL CvSeq* cvHoughLines2_P_Custom(	
	CvArr* src_image, void* lineStorage, double rho, 
	double dStartAngle, double dEndAngle, double theta, 
	int threshold, int iMinLength, int iMaxGap);