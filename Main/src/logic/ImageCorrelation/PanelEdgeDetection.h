#pragma once

//#include "Image.h"
#include "opencv\cxcore.h"
#include "opencv\cv.h"
#include "opencv\highgui.h"

#include <list>
using std::list;

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
	double dStartY;
	double dSlope;

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


private: 
	static bool PanelEdgeDetection::RobustPixelLineFit(
		const list<int>* pSetX, const list<int>* pSetY, 
		int iMaxIterations, double dMaxMeanAbsRes, 
		double* pdSlope, double* pdOffset);

	static bool PixelLineFit(
		const list<int>* pSetX, const list<int>* pSetY, 
		double* pdSlope, double* pdOffset);
};

