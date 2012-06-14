#include "EdgeDetectUtil.h"
#include <math.h>

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

	// Smooth
	IplImage* pSmoothImg = cvCreateImage(cvSize(iProcW, iProcH), IPL_DEPTH_8U, 1);
	cvSmooth(pProcImg, pSmoothImg, CV_BILATERAL, 9,9,50,50);
	//cvSaveImage("c:\\Temp\\Smooth.png", pSmoothImg);

	// Edge detection
	IplImage* pEdgeImg =  cvCreateImage(cvSize(iProcW, iProcH), IPL_DEPTH_8U, 1);
	cvCanny( pSmoothImg, pEdgeImg, 20, 10);
	//cvSaveImage("c:\\Temp\\edge.png", pEdgeImg);

	// Edge dilation for hough
	int iDilateSize = 2;
	IplConvKernel* pDilateSE = cvCreateStructuringElementEx( 
		2*iDilateSize+1, 2*iDilateSize+1, 
		iDilateSize, iDilateSize,
        CV_SHAPE_RECT);
	
	IplImage* pDilateImg =  cvCreateImage(cvSize(iProcW, iProcH), IPL_DEPTH_8U, 1);
	cvDilate( pEdgeImg, pDilateImg, pDilateSE); 
	//cvSaveImage("c:\\Temp\\DilatedEdge.png", pDilateImg);

	// Hough transform
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* lines = 0;
	// Pick potential candidate hough line
	int iThresh = (int)(ptParam->dMinLineLengthRatio * pROIImg->width/ptParam->iDecim);
	lines = cvHoughLines2(pDilateImg, storage, CV_HOUGH_PROBABILISTIC, 1, CV_PI/180/5, iThresh,  iThresh, 10);
	
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
		double dCurSize = sqrtf((double)((line[0].y-line[1].y)*(line[0].y-line[1].y)+(line[0].x-line[1].x)*(line[0].x-line[1].x)));

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

	int iNum = pSetX->size();
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
