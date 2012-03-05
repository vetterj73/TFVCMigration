#pragma once

#include "Image.h"
#include "MosaicLayer.h"
#include "JobThread.h"

using namespace MosaicDM;
#include "EdgeDetectStructDef.h"

// Wrap up job class for finding panel leading edge in a FOV
class FovPanelEdgeDetectJob : CyberJob::Job
{
public:
	FovPanelEdgeDetectJob(Image* pImage, StPanelEdgeInImage* ptParam);

	void Run();
	bool Reset();

	bool IsResultValid();

protected:
	static bool FindLeadingEdge(Image* pImage, StPanelEdgeInImage* ptParam);

private:
	Image* _pImage;
	StPanelEdgeInImage* _ptParam;
};

// Type of edge information (edge detection results)
enum EdgeInfoType
{
	INVALID,
	CONFLICTION,
	LEFTONLYVALID,
	RIGHTONLYVALID,
	BOTHVALID,
};

// Class for panel edge detection
class PanelEdgeDetection
{
public:
	PanelEdgeDetection();
	~PanelEdgeDetection(void);

	bool Initialization(MosaicLayer *pLayer, DRect panelRoi);

	FovPanelEdgeDetectJob* GetValidJob(int iLayer, int iTrig, int iCam);

	void Reset();

	EdgeInfoType CalLeadingEdgeLocation(
		double* pdSlope, double* pdLeftXOffset, double* pdRightXOffset,
		int* piLayer,int* piTrig,
		int* piLeftCam, int* piRightCam);

private:
	bool _bConveyorLeft2Right;
	bool _bConveyorFixedFrontRail;
	double _dMinLeadingEdgeGap;				// Minimum gap between image edge and nominal panel leading edge
	double _dLeadingEdgeSearchRange;		// Search range for panel leading edge
	double _dConveyorBeltAreaSize;			// The size of conveyor belt area that need to be ignored in leading edge detection
	int _iLayerIndex;
	DRect _panelRoi;
	
	int _iEdgeTrigIndex;
	int _iLeftCamIndex;
	int _iRightCamIndex; 

	PanelEdgeType _leadingEdgeType; 

	Image* _pLeftFov;
	Image* _pRightFov;

	StPanelEdgeInImage _leftFovParam;
	StPanelEdgeInImage _rightFovParam;

	FovPanelEdgeDetectJob* _pLeftFovJob;
	FovPanelEdgeDetectJob* _pRightFovJob;
};

