#include "PanelEdgeDetection.h"

PanelEdgeDetection::PanelEdgeDetection(void)
{
}


PanelEdgeDetection::~PanelEdgeDetection(void)
{
}

/*
bool PanelEdgeDetection::FindLeadingEdge(Image* pImage, StPanelEdgeInImage* ptParam)
{
	IplImage* pCvImage =  cvCreateImageHeader(cvSize(pImage->Columns(), pImage->Rows()), IPL_DEPTH_8U, 3);
	pCvImage->widthStep = pImage->ByteRowStride();
	pCvImage->imageData = (char*)pImage->GetBuffer();

	bool bFlag = FindLeadingEdge(pCvImage, ptParam);

	cvReleaseImageHeader(&pCvImage);

	return(bFlag);
}*/



bool PanelEdgeDetection::FindLeadingEdge(IplImage* pImage, StPanelEdgeInImage* ptParam)
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

	int iWidth = ptParam->iRight - ptParam->iLeft + 1;
	int iHeight = ptParam->iBottom - ptParam->iTop + 1;

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

	// Smooth
	IplImage* pSmoothImg = cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
	cvSmooth(pGrayImg, pSmoothImg, CV_BILATERAL, 9,9,50,50);
	//cvSaveImage("c:\\Temp\\Smooth.png", pSmoothImg);

	// Edge detection
	IplImage* pEdgeImg =  cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
	cvCanny( pSmoothImg, pEdgeImg, 20, 5);
	//cvSaveImage("c:\\Temp\\edge.png", pEdgeImg);

	// Edge dilation for hough
	int iDilateSize = 2;
	IplConvKernel* pDilateSE = cvCreateStructuringElementEx( 
		2*iDilateSize+1, 2*iDilateSize+1, 
		iDilateSize, iDilateSize,
        CV_SHAPE_RECT);
	
	IplImage* pDilateImg =  cvCreateImage(cvSize(iWidth, iHeight), IPL_DEPTH_8U, 1);
	cvDilate( pEdgeImg, pDilateImg, pDilateSE); 
	//cvSaveImage("c:\\Temp\\DilatedEdge.png", pDilateImg);

	IplImage* pFlipImg =  cvCreateImage(cvSize(iHeight, iWidth), IPL_DEPTH_8U, 1);
	for(int iy = 0; iy<iHeight; iy++)
	{
		for(int ix=0; ix<iWidth; ix++)
		{
			pFlipImg->imageData[(iWidth-1-ix)*pFlipImg->widthStep+iy] = pDilateImg->imageData[iy*pDilateImg->widthStep+ix];
		}
	}
	//cvSaveImage("c:\\Temp\\Flipped.png", pFlipImg);

	// Hough transform
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* lines = 0;
	int iThresh = (int)(ptParam->dMinLineLengthRatio * pROIImg->width);
	lines = cvHoughLines2(pDilateImg, storage, CV_HOUGH_STANDARD, 1, CV_PI/180, iThresh,  1, 10);

	bool bFirst = true;
	int iSelectIndex = -1;
	double dSelectRho = 0;
	for(int i = 0; i < MIN(lines->total,100); i++ )
    {
		float* line = (float*)cvGetSeqElem(lines,i);
        float rho = line[0];
        float theta = line[1];
		if(theta < CV_PI/180*(90-ptParam->dAngleRange) && 
			theta> CV_PI/180*(90+ptParam->dAngleRange))
			continue;

		if(bFirst)
		{
			iSelectIndex = i;
			dSelectRho = rho;
			bFirst = false;
		}
		else
		{
		
			if(ptParam->type == BOTTOMEDGE)
			{	// Bottom edge case
				if(dSelectRho < rho)
				{
					dSelectRho = rho;
					iSelectIndex = i;
				}
			}
			else
			{	// Top edge case
				if(dSelectRho > rho)
				{
					dSelectRho = rho;
					iSelectIndex = i;
				}
			}
		}
    }
	
	if(iSelectIndex == -1)
	{
		return(false);
	}

	float* line = (float*)cvGetSeqElem(lines,iSelectIndex);
	ptParam->dRho = line[0]+ptParam->iTop; // not very accurate
	ptParam->dTheta = line[1] - CV_PI/2;


	// for debug
    /*float rho = line[0];
    float theta = line[1];
    CvPoint pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + (pROIImg->width-100)*(-b));
    pt1.y = cvRound(y0 + (pROIImg->width-100)*(a));
    pt2.x = cvRound(x0 - (pROIImg->width-100)*(-b));
    pt2.y = cvRound(y0 - (pROIImg->width-100)*(a));
	cvLine( pROIImg, pt1, pt2, CV_RGB(255,0,0), 3, 8 );
	cvNamedWindow( "Hough", 1 );
    cvShowImage( "Hough", pROIImg );*/



	// clean up
	cvReleaseStructuringElement(&pDilateSE);
	cvReleaseImage(&pFlipImg);
	cvReleaseImage(&pDilateImg);
	cvReleaseImage(&pSmoothImg);
	cvReleaseImage(&pEdgeImg);
	if(pImage->nChannels > 1) cvReleaseImage(&pGrayImg);
	cvReleaseImageHeader(&pROIImg);

	return(true);
}
