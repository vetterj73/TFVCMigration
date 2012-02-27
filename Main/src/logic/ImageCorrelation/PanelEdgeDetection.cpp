#include "PanelEdgeDetection.h"



PanelEdgeDetection::PanelEdgeDetection(void)
{
}


PanelEdgeDetection::~PanelEdgeDetection(void)
{
}


bool PanelEdgeDetection::FindLeadingEdge(Image* pImage, StPanelEdgeInImage* ptParam)
{
	IplImage* pCvImage =  cvCreateImageHeader(cvSize(pImage->Columns(), pImage->Rows()), IPL_DEPTH_8U, 3);
	pCvImage->widthStep = pImage->ByteRowStride();
	pCvImage->imageData = (char*)pImage->GetBuffer();

	bool bFlag = ::FindLeadingEdge(pCvImage, ptParam);

	cvReleaseImageHeader(&pCvImage);

	return(bFlag);
}

