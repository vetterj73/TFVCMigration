#pragma once


#include "opencv\cv.h"
#include "opencv\cxcore.h"

CV_IMPL CvSeq* cvHoughLines2_P_Custom( CvArr* src_image, void* lineStorage, int method,
               double rho, double theta, int threshold,
               double param1, double param2 );