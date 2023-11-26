#include "EdgeDetectUtil.h"
#include <math.h>
#include <fstream>
#include "hough_custom.h"

using namespace std;

// Find panel leading edge in a FOV image
// pImage: input, color or grayscal FOV image
// ptParam: inout, control parameters and results
bool FindLeadingEdge(IplImage* pImage, StPanelEdgeInImage* ptParam)
{
	// validation check
	if (ptParam->type != TOPEDGE && ptParam->type != BOTTOMEDGE)
	{
		ptParam->iFlag = -1;
		return(false);
	}

	if(pImage == NULL || ptParam == NULL)
	{
		ptParam->iFlag = -2;
		return(false);
	}

	if( ptParam->iLeft <0 ||
		ptParam->iRight >= pImage->width ||
		ptParam->iTop <0 ||
		ptParam->iBottom >= pImage->height ||
		ptParam->iLeft >= ptParam->iRight ||
		ptParam->iTop >= ptParam->iBottom)
	{
		ptParam->iFlag = -3;
		return(false);
	}

	if(ptParam->iDecim != 1 && 
		ptParam->iDecim != 2 &&
		ptParam->iDecim != 4)
	{
		ptParam->iFlag = -4;
		return(false);
	}

	int iWidth = ptParam->iRight - ptParam->iLeft + 1;
	int iHeight = ptParam->iBottom - ptParam->iTop + 1;
	int iProcW = iWidth, iProcH = iHeight;
	if(ptParam->iDecim != 1)
	{
		iProcW = (iWidth+1)/2;
		iProcH = (iHeight+1)/2;

		if(ptParam->iDecim == 4)
		{
			iProcW = (iProcW+1)/2;
			iProcH = (iProcH+1)/2;
		}
	}

	//LogMessage("Begin prepare image");

	// ROI image
	IplImage* pROIImg =  cvCreateImageHeader(cvSize(iWidth, iHeight), IPL_DEPTH_8U, pImage->nChannels);
	pROIImg->widthStep = pImage->widthStep;
	pROIImg->imageData = (char*)pImage->imageData + pImage->widthStep*ptParam->iTop + ptParam->iLeft*pImage->nChannels;
	//cvSaveImage("c:\\Temp\\RoiImg.png", pROIImg);

	// GrayScale image
	IplImage* pGrayImg = NULL;
	if(pImage->nChannels > 1)
	{
		pGrayImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
		cvCvtColor(pROIImg, pGrayImg , CV_BGR2GRAY );
	}
	else
	{
		pGrayImg = pROIImg;
	}

	IplImage* pProcImg = pGrayImg;
	IplImage* pDecim2 = NULL, *pDecim4 = NULL;
	if(ptParam->iDecim != 1)
	{
		pDecim2 = cvCreateImage(cvSize((iWidth+1)/2, (iHeight+1)/2), IPL_DEPTH_8U, 1);
		cvPyrDown(pGrayImg, pDecim2);
		pProcImg = pDecim2;

		if(ptParam->iDecim == 4)
		{
			pDecim4 = cvCreateImage(cvSize(iProcW, iProcH), IPL_DEPTH_8U, 1);
			cvPyrDown(pDecim2, pDecim4);
			pProcImg = pDecim4;
		}
	}
	//LogMessage("End prepare image");

	//LogMessage("Begin smooth");
	// Smooth
	IplImage* pSmoothImg = cvCreateImage(cvSize(iProcW, iProcH), IPL_DEPTH_8U, 1);
	//cvSmooth(pProcImg, pSmoothImg, CV_BILATERAL, 9,9,50,50);
	jrhFastBWBilateral(pProcImg, pSmoothImg);
	//cvSaveImage("c:\\Temp\\Smooth.png", pSmoothImg);
	//LogMessage("End smooth");

	//LogMessage("Begin edge detection");
	// Edge detection
	IplImage* pEdgeImg =  cvCreateImage(cvSize(iProcW, iProcH), IPL_DEPTH_8U, 1);
	cvCanny( pSmoothImg, pEdgeImg, 20, 10);
	//cvSaveImage("c:\\Temp\\edge.png", pEdgeImg);
	//LogMessage("End edge detection");

	//LogMessage("Begin dilation");
	// Edge dilation for hough
	int iDilateSize = 1;
	IplConvKernel* pDilateSE = cvCreateStructuringElementEx( 
		2*iDilateSize+1, 2*iDilateSize+1, 
		iDilateSize, iDilateSize,
        CV_SHAPE_RECT);
	
	IplImage* pDilateImg =  cvCreateImage(cvSize(iProcW, iProcH), IPL_DEPTH_8U, 1);
	cvDilate( pEdgeImg, pDilateImg, pDilateSE); 
	//cvSaveImage("c:\\Temp\\DilatedEdge.png", pDilateImg);
	//LogMessage("End dilation");

	//LogMessage("Begin Hough");
	// Hough transform
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* lines = 0;
	// Pick potential candidate hough line
	int iThresh = (int)(ptParam->dMinLineLengthRatio * pROIImg->width/ptParam->iDecim);
	//lines = cvHoughLines2(pDilateImg, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180/5, iThresh,  iThresh, 10);
	double dStartAngle = (90 - ptParam->dAngleRange)*CV_PI/180, dEndAngle = (90 + ptParam->dAngleRange)*CV_PI/180;
	//double dStartAngle = (90-50)*CV_PI/180, dEndAngle = (90+50)*CV_PI/180;
	lines = cvHoughLines2_P_Custom(pDilateImg, storage, 1, dStartAngle, dEndAngle, CV_PI/180/10, iThresh,  iThresh, 10);
	//LogMessage("End Hough\n");

	// Pick the right hough lines
	bool bFirst = true;
	int iSelectIndex = -1;
	double dSelectY = -1;
	if(ptParam->type == TOPEDGE) dSelectY = 1e5;
	double dSelectSize = -1;
	double dMaxSlope = tan(ptParam->dAngleRange);
	for(int i = 0; i < MIN(lines->total,100); i++ )
    {
		// Get a line
		CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
	
		// Validation check
		if(abs(line[1].x-line[0].x) < 10)
			continue;

		double slope = ((double)line[1].y-(double)line[0].y)/ ((double)line[1].x-(double)line[0].x);
		if(fabs(slope)<dMaxSlope)
			continue;

		// Current line information
		double dCurY = (line[0].y+line[1].y)/2.;
		double dCurSize = sqrt((double)((line[0].y-line[1].y)*(line[0].y-line[1].y)+(line[0].x-line[1].x)*(line[0].x-line[1].x)));

		if(ptParam->type == BOTTOMEDGE) // For leading edge on the bottom of panel
		{	// Y is much bigger or Y is similar but size is bigger
			if((dCurY -dSelectY) > iDilateSize*2 ||
				(fabs(dCurY -dSelectY) <= iDilateSize*2 && dCurSize > dSelectSize))
			{
				dSelectY = dCurY;
				dSelectSize = dCurSize;
				iSelectIndex = i;
			}
		}
		else	// For leading edge on the top of panel 
		{	// Y is much samller or Y is similar but size is bigger
			if((dCurY -dSelectY) < iDilateSize*2 ||
				(fabs(dCurY -dSelectY) <= iDilateSize*2 && dCurSize > dSelectSize))
			{
				dSelectY = dCurY;
				dSelectSize = dCurSize;
				iSelectIndex = i;
			}
		}
    }
	
	// No leading ede is detected
	if(iSelectIndex == -1)
	{
		ptParam->iFlag = -10;
		return(false);
	}

	// The selected hogh line
	CvPoint* line = (CvPoint*)cvGetSeqElem(lines,iSelectIndex);
	double dSlope = ((double)line[1].y-(double)line[0].y)/ ((double)line[1].x-(double)line[0].x);
	
	// Collect valid edge pixels
	unsigned  char* pBuf =(unsigned char*) pEdgeImg->imageData;
	int iStep = pEdgeImg->widthStep;
	list<int> iSetX, iSetY;
	for(int ix = line[0].x; ix <= line[1].x; ix++)
	{
		int iyLine = (int)(line[0].y+  dSlope*(ix-line[0].x) + 0.5);
		int	iyMin = iyLine -2*iDilateSize;
		if(iyMin < 0) iyMin = 0;
		int iyMax = iyLine + 2*iDilateSize;
		if(iyMax > iHeight-1) iyMax = iHeight-1;
		for(int iy = iyMin; iy<=iyMax; iy++)
		{
			if(pBuf[iy*iStep+ix]>0)
			{
				iSetX.push_back(ix);
				iSetY.push_back(iy);
			}
		}
	}

	// Robust line fit
	double dOffset;
	bool bFlag = RobustPixelLineFit(
		&iSetX, &iSetY, 
		4, 1.0, 
		&dSlope, &dOffset);
	
	// Convert for image origin (top left corner)
	ptParam->dSlope = dSlope;
	ptParam->dRowOffsetInColumn0 = (dOffset - ptParam->dSlope*ptParam->iLeft)*ptParam->iDecim +ptParam->iTop;

	// clean up
	cvReleaseStructuringElement(&pDilateSE);
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&pDilateImg);
	cvReleaseImage(&pSmoothImg);
	cvReleaseImage(&pEdgeImg);
	if(pDecim2 != NULL) cvReleaseImage(&pDecim2);
	if(pDecim4 != NULL) cvReleaseImage(&pDecim4);
	if(pImage->nChannels > 1) cvReleaseImage(&pGrayImg);
	cvReleaseImageHeader(&pROIImg);


	ptParam->iFlag = 1;
	return(true);
}

// Robust fit a line (Calcualte line slop and offset)
// pSetX and pSetY: input, pixel locations
// iMaxIterations: maximum loop iterations
// dMaxMeanAbsRes: return results if mean(abs(residual)) < dMaxMeanAbsRes
// pdSlope: output, line slope and offset
bool RobustPixelLineFit(
	const list<int>* pSetX, const list<int>* pSetY, 
	int iMaxIterations, double dMaxMeanAbsRes, 
	double* pdSlope, double* pdOffset)
{
	// Validation Check
	if(pSetX->size() < 2 || 
		pSetY->size()<2 || 
		pSetX->size() != pSetY->size())
		return(false);

	list<int> setX_in, setY_in;
	list<int> setX_out, setY_out;
	// Input data copy
	list<int>::const_iterator i, j;
	for(i = pSetX->begin(), j = pSetY->begin(); i!=pSetX->end(); i++, j++)
	{
		setX_in.push_back(*i);
		setY_in.push_back(*j);
	}

	for(int k=0; k<iMaxIterations; k++)
	{	
		// Line fit
		double dSlope, dOffset;
		bool bFlag = PixelLineFit(&setX_in, &setY_in, &dSlope, &dOffset);
		if(!bFlag) return(false);

		// Calcualte mean and sdv
		int iCount = 0;
		double dSum = 0;
		double dSumSq = 0;
		for(i = setX_in.begin(), j = setY_in.begin(); i!=setX_in.end(); i++, j++)
		{
			double dRes = fabs((*i)*dSlope+dOffset -(*j));
			dSum += dRes;
			dSumSq += dRes*dRes;
			iCount++;
		}
		double dMean = dSum/iCount;
		double dSdv = sqrt(dSumSq/iCount - dMean*dMean);
		
		// Terminate condition
		if(dMean < dMaxMeanAbsRes ||
			k == iMaxIterations-1)
		{
			*pdSlope = dSlope;
			*pdOffset = dOffset;
			return(true);
		}

		// Reduce pixels based on residual
		setX_out.clear();
		setY_out.clear();
		for(i = setX_in.begin(), j = setY_in.begin(); i!=setX_in.end(); i++, j++)
		{
			double dRes = fabs((*i)*dSlope+dOffset -(*j));
			if(dRes < dMean + 0.5*dSdv)
			{
				setX_out.push_back(*i);
				setY_out.push_back(*j);
			}
		}
		
		setX_in.clear();
		setY_in.clear();
		for(i = setX_out.begin(), j = setY_out.begin(); i!=setX_out.end(); i++, j++)
		{
			setX_in.push_back(*i);
			setY_in.push_back(*j);
		}
	}

	// Should never reach here
	return(false);
}

// Least square fit a line (Calcualte line slop and offset)
// pSetX and pSetY: input, pixel locations
// pdSlope: output, line slope and offset
// http://easycalculation.com/statistics/learn-regression.php
bool PixelLineFit(
	const list<int>* pSetX, const list<int>* pSetY, 
	double* pdSlope, double* pdOffset)
{
	// Validation Check
	if(pSetX->size() < 2 || 
		pSetY->size()<2 || 
		pSetX->size() != pSetY->size())
		return(false);

	int iNum = (int)pSetX->size();
	double sum_X=0, sum_Y=0, sum_XY=0, sum_X2=0;
	list<int>::const_iterator i, j;
	for(i = pSetX->begin(), j = pSetY->begin(); i!=pSetX->end(); i++, j++)
	{
		sum_X += *i;
		sum_Y += *j;
		sum_XY += (*i)*(*j);
		sum_X2 += (*i)*(*i);
	}

	*pdSlope= (iNum*sum_XY-sum_X*sum_Y)/(iNum*sum_X2-sum_X*sum_X);
	*pdOffset = (sum_Y-(*pdSlope)*sum_X)/iNum;

	return(true);
}


// For debug
void LogMessage(char* pMessage)
{
	SYSTEMTIME st;
	GetLocalTime(&st);
	
	ofstream of("C:\\Temp\\PanelDetectionUnitTestLog.txt", ios_base::app);

	of << st.wHour << ":"
		<< st.wMinute << ":" 
		<< st.wSecond << ":" 
		<< st.wMilliseconds << "::" 
		<< pMessage
		<< endl;  

	of.close();
}

#pragma region JH grayscale fast bilateral filter

float color_weight[] = {
1.000000f,
0.999800f,
0.999200f,
0.998202f,
0.996805f,
0.995012f,
0.992826f,
0.990248f,
0.987282f,
0.983931f,
0.980199f,
0.976090f,
0.971611f,
0.966765f,
0.961558f,
0.955997f,
0.950089f,
0.943839f,
0.937255f,
0.930345f,
0.923116f,
0.915578f,
0.907738f,
0.899605f,
0.891188f,
0.882497f,
0.873541f,
0.864331f,
0.854875f,
0.845185f,
0.835270f,
0.825142f,
0.814810f,
0.804286f,
0.793581f,
0.782705f,
0.771669f,
0.760484f,
0.749162f,
0.737713f,
0.726149f,
0.714480f,
0.702718f,
0.690872f,
0.678955f,
0.666977f,
0.654948f,
0.642878f,
0.630779f,
0.618660f,
0.606531f,
0.594402f,
0.582282f,
0.570182f,
0.558110f,
0.546074f,
0.534085f,
0.522150f,
0.510278f,
0.498476f,
0.486752f,
0.475114f,
0.463569f,
0.452123f,
0.440784f,
0.429557f,
0.418449f,
0.407465f,
0.396611f,
0.385891f,
0.375311f,
0.364875f,
0.354588f,
0.344452f,
0.334473f,
0.324652f,
0.314995f,
0.305502f,
0.296176f,
0.287021f,
0.278037f,
0.269227f,
0.260592f,
0.252133f,
0.243850f,
0.235746f,
0.227820f,
0.220072f,
0.212503f,
0.205112f,
0.197899f,
0.190863f,
0.184004f,
0.177320f,
0.170811f,
0.164474f,
0.158310f,
0.152316f,
0.146490f,
0.140830f,
0.135335f,
0.130003f,
0.124830f,
0.119816f,
0.114957f,
0.110251f,
0.105695f,
0.101287f,
0.097024f,
0.092903f,
0.088922f,
0.085077f,
0.081366f,
0.077786f,
0.074333f,
0.071005f,
0.067800f,
0.064713f,
0.061741f,
0.058883f,
0.056135f,
0.053493f,
0.050956f,
0.048519f,
0.046180f,
0.043937f,
0.041786f,
0.039724f,
0.037749f,
0.035858f,
0.034047f,
0.032316f,
0.030660f,
0.029077f,
0.027565f,
0.026121f,
0.024743f,
0.023429f,
0.022175f,
0.020980f,
0.019841f,
0.018757f,
0.017725f,
0.016743f,
0.015809f,
0.014921f,
0.014077f,
0.013276f,
0.012515f,
0.011794f,
0.011109f,
0.010460f,
0.009845f,
0.009262f,
0.008711f,
0.008189f,
0.007695f,
0.007228f,
0.006787f,
0.006370f,
0.005976f,
0.005604f,
0.005254f,
0.004923f,
0.004612f,
0.004318f,
0.004041f,
0.003781f,
0.003536f,
0.003305f,
0.003089f,
0.002885f,
0.002694f,
0.002514f,
0.002346f,
0.002187f,
0.002039f,
0.001900f,
0.001770f,
0.001648f,
0.001534f,
0.001427f,
0.001327f,
0.001234f,
0.001146f,
0.001065f,
0.000989f,
0.000918f,
0.000851f,
0.000789f,
0.000732f,
0.000678f,
0.000628f,
0.000582f,
0.000538f,
0.000498f,
0.000460f,
0.000426f,
0.000393f,
0.000363f,
0.000335f,
0.000310f,
0.000286f,
0.000263f,
0.000243f,
0.000224f,
0.000206f,
0.000190f,
0.000175f,
0.000161f,
0.000148f,
0.000136f,
0.000125f,
0.000115f,
0.000105f,
0.000097f,
0.000089f,
0.000081f,
0.000074f,
0.000068f,
0.000063f,
0.000057f,
0.000052f,
0.000048f,
0.000044f,
0.000040f,
0.000037f,
0.000033f,
0.000031f,
0.000028f,
0.000025f,
0.000023f,
0.000021f,
0.000019f,
0.000018f,
0.000016f,
0.000015f,
0.000013f,
0.000012f,
0.000011f,
0.000010f,
0.000009f,
0.000008f,
0.000007f,
0.000007f,
0.000006f,
0.000006f,
0.000005f,
0.000005f,
0.000004f,
0.000004f,
0.000003f,
0.000003f,
0.000003f,
0.000002f,
0.000002f,
0.000002f,
0.000002f,
0.000002f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000001f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f,
0.000000f
};



void bilateralFilterBWPoint_8u(uchar* sptr, int stride, uchar* result)
{
	int radius = 3;

    int d = radius*2 + 1;
    int di, dj;

	float sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
    int b0 = sptr[0];

	for ( di = -radius; di <= radius; di++)
	{
		for (dj = -radius; dj <= radius; dj++)
		{
	        int b = sptr[di*stride+dj];
	        float w = color_weight[std::abs(b - b0)];
	        sum_b += b*w;
		    wsum += w;
         }
	}
    result[0] = cvRound(sum_b/wsum); 
}

void jrhFastBWBilateral(IplImage* img, IplImage* bf) 
{
	const int radius = 3;

	// create a padded copy of the input, to simplify code.  
	// Performance penalty is minimal, code readability is improved.
	IplImage* tmpImg = cvCreateImage(cvSize(img->width+2*radius+2, img->height+2*radius+2), img->depth, img->nChannels);
	cvCopyMakeBorder( img, tmpImg, cvPoint(radius+1,radius+1),IPL_BORDER_REPLICATE);

	// width = ncols, height = nrows.
	const int ncols = img->width;
	const int nrows = img->height;

	vector<int> _csums(ncols+(2*radius+1));
    int* csums = &_csums[(radius+1)];

	vector<int> _csum2s(ncols+(2*radius+1));
    int* csum2s = &_csum2s[(radius+1)];

	double normalizer = 1.0 / (2*radius + 1) / (2*radius+1);

   int stride = tmpImg->widthStep;
   uchar* data = (uchar *) &tmpImg->imageData[(radius+1)*stride+(radius+1)];

	for (int col = -(radius+1); col < (ncols+radius); col ++) 
	{
		csums[col] = 0;
		csum2s[col] = 0;
		for (int row = -(radius+1); row < radius; row++)
		{
			csums[col]    += data[row*stride+col];
			csum2s[col]    += data[row*stride+col]  *data[row*stride+col];
		}
	}

   // from here on out, row/col indexes refer to the row/col index
   // in the destination image.
   // dst(col,row) = img(col,row) = tmpImg(col+4,row+4)
   //                 colums uses same indexing as img, so cols[col] gives the colsum of column col in img.
	for (int row = 0; row < nrows; row++) 
	{
	   int asum0 = 0;
	   int a2sum0 = 0;
		for (int col = -(radius+1); col < radius; col++) 
		{
			csums[col]   += data[(row+radius)*stride+col]   - data[(row-radius-1)*stride+col];
			csum2s[col]   += data[(row+radius)*stride+col]  *data[(row+radius)*stride+col]   - data[(row-radius-1)*stride+col]  *data[(row-radius-1)*stride+col];
			asum0        += csums[col];
			a2sum0        += csum2s[col];

		}
		for (int col = 0; col < ncols; col++)
		{
			csums[col+radius]     += data[(row+radius)*stride+col+radius]   - data[(row-radius-1)*stride+col+radius];
			csum2s[col+radius]    += data[(row+radius)*stride+col+radius]   * data[(row + radius)*stride+col+radius]   - data[(row-radius-1)*stride+col+radius]   * data[(row-radius-1)*stride+col+radius];
			asum0  += csums [col+radius] - csums [col-(radius+1)];
			a2sum0 += csum2s[col+radius] - csum2s[col-(radius+1)];

			bf->imageData[row*bf->widthStep+col]   = cvRound(asum0*normalizer);

			double bVar = a2sum0*normalizer-(asum0*normalizer)*(asum0*normalizer);

			if (bVar > 80)
			{
				bilateralFilterBWPoint_8u(& data[row*stride+col], stride, &((uchar*)bf->imageData)[row*bf->widthStep+col]);
			}
		}
	}
	cvReleaseImage(&tmpImg);
}

#pragma endregion
