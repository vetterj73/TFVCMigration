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

	/* For debug
	float dSlope = ptParam->dSlope;
	float dStartRow = ptParam->dStartRow;
	CvPoint pt1, pt2;
	pt1.x = 0;
	pt1.y = dStartRow;
	pt2.x = 2500;
	pt2.y = pt1.y +dSlope*pt2.x;
	cvLine( pCvImage, pt1, pt2, CV_RGB(255,255,255), 1, 8 );
	if(ptParam->iLeft >10)
	{
		cvSaveImage("C:\\Temp\\edgeImage_left.png", pCvImage);
	}
	else
	{
		cvSaveImage("C:\\Temp\\edgeImage_right.png", pCvImage);
	}
	//*/
	
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
// pdSlope: out, the slope of leading edge (delta_x/delta_Y)
// pdLeftXOffset and pdRightXOffset: out, the x offsets of left and right FOVs
// piLayer, piTrig, piLeftCam and piRightCam: out, the indics of left and right FOVs
EdgeInfoType PanelEdgeDetection::CalLeadingEdgeLocation(
	double* pdSlope, double* pdLeftXOffset, double* pdRightXOffset,
	int* piLayer,int* piTrig,
	int* piLeftCam, int* piRightCam)
{
	// Fov indexes
	*piLayer = _iLayerIndex;
	*piTrig = _iEdgeTrigIndex;
	*piLeftCam = _iLeftCamIndex; 
	*piRightCam = _iRightCamIndex;

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
		return(INVALID);
	}
	
	// Only left result is valid
	if(bLeftValid && !bRightValid)
	{
		*pdSlope = _leftFovParam.dSlope;   // delta_x/delta_Y
		double dAngle = atan(*pdSlope); 
		*pdLeftXOffset = -_leftFovParam.dStartRow*cos(dAngle)*_pLeftFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			*pdLeftXOffset += _panelRoi.xMax;
		return(LEFTONLYVALID);
	}

	// Only right result is valid
	if(!bLeftValid && bRightValid)
	{
		*pdSlope = _rightFovParam.dSlope;   // delta_x/delta_Y
		double dAngle = atan(*pdSlope); 
		*pdRightXOffset = -_rightFovParam.dStartRow*cos(dAngle)*_pRightFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			*pdRightXOffset += _panelRoi.xMax;
		return(RIGHTONLYVALID);
	}

	// Both results are valid
	if(bLeftValid && bRightValid)
	{
		// Calculat slope of panel leading edge
		// Assume board is rigid and calibration is almost consistent
		// So that norminal transforms can be used in the calculation
			// Center point of edge in FOV
		double dLeftRoiCenCols = (_leftFovParam.iLeft + _leftFovParam.iRight)/2.0;
		double dLeftRoiCenRows = _leftFovParam.dStartRow + _leftFovParam.dSlope*dLeftRoiCenCols;
		double dLeftCenX, dLeftCenY;
		_pLeftFov->ImageToWorld(dLeftRoiCenRows, dLeftRoiCenCols, &dLeftCenX, &dLeftCenY); 

		double dRightRoiCenCols = (_rightFovParam.iLeft + _rightFovParam.iRight)/2;
		double dRightRoiCenRows = _rightFovParam.dStartRow + _rightFovParam.dSlope*dRightRoiCenCols;
		double dRightCenX, dRightCenY;
		_pRightFov->ImageToWorld(dRightRoiCenRows, dRightRoiCenCols, &dRightCenX, &dRightCenY); 
		
			// Panel edge slope based on both results
		double dPanelEdgeSlope = (dRightCenX-dLeftCenX)/(dRightCenY-dLeftCenY);

		// Slope Consistent check
		if(fabs(dPanelEdgeSlope-_leftFovParam.dSlope) > tan(PI/180) ||
			fabs(dPanelEdgeSlope-_rightFovParam.dSlope) > tan(PI/180))
		{
			return(CONFLICTION);
		}

		*pdSlope = dPanelEdgeSlope;	
		double dAngle = atan(*pdSlope); 

		// Left x offset
		double dStartRow = dLeftRoiCenRows - *pdSlope*dLeftRoiCenCols; // updated dStartRow
		*pdLeftXOffset = -dStartRow*cos(dAngle)*_pLeftFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			*pdLeftXOffset += _panelRoi.xMax;
		
		// Right x offset
		dStartRow = dRightRoiCenRows - *pdSlope*dRightRoiCenCols;
		*pdRightXOffset = -dStartRow*cos(dAngle)*_pRightFov->PixelSizeY();
		if(_leadingEdgeType == BOTTOMEDGE) 
			*pdRightXOffset += _panelRoi.xMax;

		return(BOTHVALID);
	}

	// Should never reach here 
	return(INVALID);
}
#pragma endregion



