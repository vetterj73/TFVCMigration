#include "PanelEdgeDetection.h"
#include "CorrelationParameters.h"
#include "EdgeDetectUtil.h"

#include "ColorImage.h"
#include <math.h>

#define PI 3.14159

#pragma region FovPanelEdgeDetectJob
FovPanelEdgeDetectJob::FovPanelEdgeDetectJob(Image* pImage, StPanelEdgeInImage* ptParam)
{
	_pImage = pImage;
	_ptParam = ptParam;
}

void FovPanelEdgeDetectJob::Run()
{
	FindLeadingEdge(_pImage, _ptParam);
}

bool FovPanelEdgeDetectJob::Reset()
{
	_ptParam->Reset();

	return(true);
}

bool FovPanelEdgeDetectJob::IsResultValid()
{
	if(_ptParam->iFlag > 0)
		return(true);
	else
		return(false);
}

bool FovPanelEdgeDetectJob::FindLeadingEdge(Image* pImage, StPanelEdgeInImage* ptParam)
{
	Image* pProcImage = new Image(*pImage);
	IplImage* pCvImage =NULL;
	// If this is color image and (not YCrCb or seperatly stored channel)
	if(pImage->GetBytesPerPixel() > 1 && 
		(!((ColorImage*)pImage)->IsChannelStoredSeperate() || ((ColorImage*)pImage)->GetColorStyle() != YCrCb ))
	{
		if(!pProcImage->HasOwnBuffer())
		{
			pProcImage->CreateOwnBuffer();
			::memcpy(pProcImage->GetBuffer(), pImage->GetBuffer(), pImage->BufferSizeInBytes());
		}
		((ColorImage*)pProcImage)->SetChannelStoreSeperated(false);
		((ColorImage*)pProcImage)->SetColorStyle(BGR);
		pCvImage =  cvCreateImageHeader(cvSize(pProcImage->Columns(), pProcImage->Rows()), IPL_DEPTH_8U, 3);
		pCvImage->widthStep = pProcImage->PixelRowStride()*3;
	}
	else
	{
		pCvImage =  cvCreateImageHeader(cvSize(pProcImage->Columns(), pProcImage->Rows()), IPL_DEPTH_8U, 1);
		pCvImage->widthStep = pProcImage->PixelRowStride();
	}
	pCvImage->imageData = (char*)pProcImage->GetBuffer();

	bool bFlag = ::FindLeadingEdge(pCvImage, ptParam);

	// For debug
	if(CorrelationParametersInst.bSavePanelEdgeDebugImages)
	{
		float dSlope = ptParam->dSlope;
		float dRowOffsetInColumn0 = ptParam->dRowOffsetInColumn0;
		CvPoint pt1, pt2;
		pt1.x = 0;
		pt1.y = dRowOffsetInColumn0;
		pt2.x = 2500;
		pt2.y = pt1.y +dSlope*pt2.x;
		cvLine( pCvImage, pt1, pt2, CV_RGB(255,255,255), 1, 8 );
		
		char cTemp[100];
				
		if(ptParam->iLeft >10)
			sprintf_s(cTemp, 100, "%sedgeImage_left.png", CorrelationParametersInst.sDiagnosticPath.c_str());		
		else
			sprintf_s(cTemp, 100, "%sedgeImage_right.png", CorrelationParametersInst.sDiagnosticPath.c_str());
		
		cvSaveImage(cTemp, pCvImage);
	}
	
	cvReleaseImageHeader(&pCvImage);

	delete pProcImage;

	return(bFlag);
}
#pragma endregion

#pragma region PanelEdgeDetection
PanelEdgeDetection::PanelEdgeDetection()
{
	_bConveyorLeft2Right =	CorrelationParametersInst.bConveyorLeft2Right;
	_bConveyorFixedFrontRail = CorrelationParametersInst.bConveyorFixedFrontRail;
	_dMinLeadingEdgeGap = CorrelationParametersInst.dMinLeadingEdgeGap;				
	_dLeadingEdgeSearchRange = CorrelationParametersInst.dLeadingEdgeSearchRange;
	_dConveyorBeltAreaSize = CorrelationParametersInst.dConveyorBeltAreaSize;

	_iLayerIndex = -1;
	_iEdgeTrigIndex = -1;
	_iLeftCamIndex = -1;
	_iRightCamIndex = -1;

	_pLeftFov = NULL;
	_pRightFov = NULL;

	_pLeftFovJob = NULL;
	_pRightFovJob = NULL;
}


PanelEdgeDetection::~PanelEdgeDetection(void)
{
	if(_pLeftFovJob != NULL) 
		delete _pLeftFovJob;
	if(_pRightFovJob != NULL) 
		delete _pRightFovJob;

	_pLeftFovJob = NULL;
	_pRightFovJob = NULL;
}


bool PanelEdgeDetection::Initialization(MosaicLayer *pLayer, DRect panelRoi)
{
	_panelRoi = panelRoi;

	// Index of layers
	_iLayerIndex = pLayer->Index();

	if(_bConveyorLeft2Right)
	{	
		// conveyor moving left to right
		_iEdgeTrigIndex = 0;
		_leadingEdgeType = BOTTOMEDGE;
	}
	else
	{	
		// Conveyor moving right to left
		_iEdgeTrigIndex = pLayer->GetNumberOfTriggers() - 1;
		_leadingEdgeType = TOPEDGE;
	}
		
	// Validation check (enough gap between nominal panel edge and FOV edge)
	Image* pImage = pLayer->GetImage(0, _iEdgeTrigIndex);
	DRect fovRect = pImage->GetBoundBoxInWorld();
	if(_bConveyorLeft2Right)
	{	
		// conveyor moving left to right
		if(fovRect.xMax < panelRoi.xMax + _dMinLeadingEdgeGap)
			return(false);
	}
	else
	{	
		// Conveyor moving right to left
		if(fovRect.xMin > panelRoi.xMin - _dMinLeadingEdgeGap)
			return(false);
	}

	int iNumCam = pLayer->GetNumberOfCameras();		
	
	// Panel edge nominal position for detection
	double dPanelEdgeInX = panelRoi.xMax;
	if(_leadingEdgeType == TOPEDGE)
		dPanelEdgeInX = panelRoi.xMin;

	// Left and right Belt edge 
	double dLeftBeltEdge = panelRoi.yMin + _dConveyorBeltAreaSize;
	double dRightBeltEdge = panelRoi.yMax - _dConveyorBeltAreaSize; 
	
	// Left FOV select and setup
	for(int iCam=0; iCam<iNumCam; iCam++)
	{
		Image* pImage = pLayer->GetImage(iCam, _iEdgeTrigIndex);
		DRect fovRect = pImage->GetBoundBoxInWorld();

		// Panel edge size suitable for detection
		double dSize = fovRect.yMax- dLeftBeltEdge;
		// If size is big enough 
		if(dSize > pImage->PixelSizeX()*pImage->Columns()/2)
		{
			_iLeftCamIndex =  iCam;

			// Edge Search range in X
			double dMinX = dPanelEdgeInX - _dLeadingEdgeSearchRange;
			if(dMinX < fovRect.xMin) dMinX = fovRect.xMin;
			double dMaxX = dPanelEdgeInX + _dLeadingEdgeSearchRange;
			if(dMaxX > fovRect.xMax) dMaxX = fovRect.xMax;

			//Edge serach range in Y
				// Bigger one of belt left edge, or fov left edge
			double dMinY = dLeftBeltEdge > fovRect.yMin ? dLeftBeltEdge : fovRect.yMin;
				// Smaller one of belt right edge or fov right edge 
			double dMaxY = dRightBeltEdge < fovRect.yMax ? dRightBeltEdge : fovRect.yMax;

			// Convert (x,y) to (row, column)
			double dTop, dBottom, dLeft, dRight;
			pImage->WorldToImage(dMinX, dMinY, &dTop, &dLeft);
			pImage->WorldToImage(dMaxX, dMaxY, &dBottom, &dRight);

			// Create edge detection job
			_leftFovParam.iLeft = (int)dLeft + 2;
			_leftFovParam.iTop = (int)dTop + 2;
			_leftFovParam.iRight = (int)dRight - 2;
			_leftFovParam.iBottom = (int)dBottom -2;
			_leftFovParam.type = _leadingEdgeType;

			_pLeftFov = pImage;
			_pLeftFovJob = new FovPanelEdgeDetectJob(pImage, &_leftFovParam);

			break;
		}
	}

	// Right FOV select and setup
	for(int iCam = iNumCam-1; iCam >=0; iCam--)
	{
		Image* pImage = pLayer->GetImage(iCam, _iEdgeTrigIndex);
		DRect fovRect = pImage->GetBoundBoxInWorld();

		// Panel edge size suitable for detection
		double dSize = dRightBeltEdge - fovRect.yMin;
		// If size is big enough
		if(dSize > pImage->PixelSizeX()*pImage->Columns()/2)
		{
			// If the left camera and right one are the same 
			if(iCam <= _iLeftCamIndex)
				break;

			_iRightCamIndex =  iCam;
		
			// Edge Search range in X
			double dMinX = dPanelEdgeInX - _dLeadingEdgeSearchRange;
			if(dMinX < fovRect.xMin) dMinX = fovRect.xMin;
			double dMaxX = dPanelEdgeInX + _dLeadingEdgeSearchRange;
			if(dMaxX > fovRect.xMax) dMaxX = fovRect.xMax;

			//Edge serach range in Y
				// Bigger one of belt left edge, or fov left edge
			double dMinY = dLeftBeltEdge > fovRect.yMin ? dLeftBeltEdge : fovRect.yMin;
				// Smaller one of belt right edge or fov right edge 
			double dMaxY = dRightBeltEdge < fovRect.yMax ? dRightBeltEdge : fovRect.yMax;

			// Convert (x,y) to (row, column)
			double dTop, dBottom, dLeft, dRight;
			pImage->WorldToImage(dMinX, dMinY, &dTop, &dLeft);
			pImage->WorldToImage(dMaxX, dMaxY, &dBottom, &dRight);

			// Create edge detection job
			_rightFovParam.iLeft = (int)dLeft + 2;
			_rightFovParam.iTop = (int)dTop + 2;
			_rightFovParam.iRight = (int)dRight - 2;
			_rightFovParam.iBottom = (int)dBottom -2;
			_rightFovParam.type = _leadingEdgeType;

			_pRightFov = pImage;
			_pRightFovJob = new FovPanelEdgeDetectJob(pImage, &_rightFovParam);

			break;
		}
	}

	return(true);
}

void PanelEdgeDetection::Reset()
{
	if(_pLeftFovJob != NULL) 
		_pLeftFovJob->Reset();
	if(_pRightFovJob != NULL) 
		_pRightFovJob->Reset();
}

// Get job for certain FOV
FovPanelEdgeDetectJob* PanelEdgeDetection::GetValidJob(int iLayer, int iTrig, int iCam)
{
	// If there is panel edge detection job with this FOV
	if(iLayer == _iLayerIndex &&
		iTrig == _iEdgeTrigIndex &&
		(iCam == _iLeftCamIndex ||
		iCam == _iRightCamIndex))
	{
		if(iCam == _iLeftCamIndex)
			return(_pLeftFovJob);
		else if(iCam == _iRightCamIndex)
			return(_pRightFovJob);
	}
	
	// If there is no panel edge detection job with this FOV 
	return(NULL);
}

// Calculate location of panel leading edge
// pInfo: out, panel leading edge information
// Return false if there is a fatal error
bool PanelEdgeDetection::CalLeadingEdgeLocation(EdgeInfo* pInfo)
{
	// Fov indexes
	pInfo->iLayerIndex = _iLayerIndex;
	pInfo->iTrigIndex = _iEdgeTrigIndex;
	pInfo->iLeftCamIndex = _iLeftCamIndex; 
	pInfo->iRightCamIndex = _iRightCamIndex;

	// Check the validataion of the results
	bool bLeftValid = false, bRightValid = false;
	if(_pLeftFovJob != NULL)
		if(_pLeftFovJob->IsResultValid())
			bLeftValid= true;

	if(_pRightFovJob != NULL)
		if(_pRightFovJob->IsResultValid())
			bRightValid= true;

	// Both results are not valid
	if(!bLeftValid && !bRightValid)
	{
		pInfo->type = INVALID;
		return(true);
	}
	
	// Only left result is valid
	if(bLeftValid && !bRightValid)
	{
		pInfo->dPanelSlope = _leftFovParam.dSlope;   // delta_x/delta_Y
		double dAngle = atan(pInfo->dPanelSlope); 
		pInfo->dLeftXOffset = -_leftFovParam.dRowOffsetInColumn0*cos(dAngle)*_pLeftFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			pInfo->dLeftXOffset += _panelRoi.xMax;

		pInfo->type = LEFTONLYVALID;
		return(true);
	}

	// Only right result is valid
	if(!bLeftValid && bRightValid)
	{
		pInfo->dPanelSlope = _rightFovParam.dSlope;   // delta_x/delta_Y
		double dAngle = atan(pInfo->dPanelSlope); 
		pInfo->dRightXOffset = -_rightFovParam.dRowOffsetInColumn0*cos(dAngle)*_pRightFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			pInfo->dRightXOffset += _panelRoi.xMax;

		pInfo->type = RIGHTONLYVALID;
		return(true);
	}

	// Both results are valid
	if(bLeftValid && bRightValid)
	{
		// Calculat slope of panel leading edge
		// Assume board is rigid and calibration is almost consistent
		// So that norminal transforms can be used in the calculation
			// Center point of edge in FOV
		double dLeftRoiCenCols = (_leftFovParam.iLeft + _leftFovParam.iRight)/2.0;
		double dLeftRoiCenRows = _leftFovParam.dRowOffsetInColumn0 + _leftFovParam.dSlope*dLeftRoiCenCols;
		double dLeftCenX, dLeftCenY;
		_pLeftFov->ImageToWorld(dLeftRoiCenRows, dLeftRoiCenCols, &dLeftCenX, &dLeftCenY); 

		double dRightRoiCenCols = (_rightFovParam.iLeft + _rightFovParam.iRight)/2;
		double dRightRoiCenRows = _rightFovParam.dRowOffsetInColumn0 + _rightFovParam.dSlope*dRightRoiCenCols;
		double dRightCenX, dRightCenY;
		_pRightFov->ImageToWorld(dRightRoiCenRows, dRightRoiCenCols, &dRightCenX, &dRightCenY); 
		
			// Panel edge slope based on both results
		double dPanelEdgeSlope = (dRightCenX-dLeftCenX)/(dRightCenY-dLeftCenY);

		// Slope Consistent check
		if(fabs(dPanelEdgeSlope-_leftFovParam.dSlope) > tan(PI/180) ||
			fabs(dPanelEdgeSlope-_rightFovParam.dSlope) > tan(PI/180))
		{
			pInfo->type = CONFLICTION;
			return(true);
		}

		pInfo->dPanelSlope = dPanelEdgeSlope;	
		double dAngle = atan(pInfo->dPanelSlope); 

		// Left x offset
		double dRowOffsetInColumn0 = dLeftRoiCenRows - pInfo->dPanelSlope*dLeftRoiCenCols; // updated dRowOffsetInColumn0
		pInfo->dLeftXOffset = -dRowOffsetInColumn0*cos(dAngle)*_pLeftFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			pInfo->dLeftXOffset += _panelRoi.xMax;
		
		// Right x offset
		dRowOffsetInColumn0 = dRightRoiCenRows - pInfo->dPanelSlope*dRightRoiCenCols;
		pInfo->dRightXOffset = -dRowOffsetInColumn0*cos(dAngle)*_pRightFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			pInfo->dRightXOffset += _panelRoi.xMax;

		pInfo->type = BOTHVALID;
		return(true);
	}

	// Should never reach here 
	return(false);
}
#pragma endregion



