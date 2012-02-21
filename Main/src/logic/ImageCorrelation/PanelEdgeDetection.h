#pragma once

//#include "Image.h"

#include "opencv\cxcore.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"

enum PanelEdgeType
{
	LEFTEDGE,
	RIGHTEDGE,
	TOPEDGE,
	BOTTOMEDGE
};

struct StPanelEdgeInImage
{
	int iLeft;
	int iRight;
	int iTop;
	int iBottom;
	PanelEdgeType type;
	double dMinLineLengthRatio;
	double dAngleRange;
	int iFlag;	
	double dRho;
	double dTheta;

	StPanelEdgeInImage()
	{
		dMinLineLengthRatio =0.5;
		dAngleRange = 3;
		iFlag = 0;
	}
};


class PanelEdgeDetection
{
public:
	PanelEdgeDetection(void);
	~PanelEdgeDetection(void);

	//static bool FindLeadingEdge(Image* pImage, StPanelEdgeInImage* ptParam);
	static bool FindLeadingEdge(IplImage* pImage, StPanelEdgeInImage* ptParam);
};

