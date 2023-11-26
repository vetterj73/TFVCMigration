#pragma once

enum PanelEdgeType
{
	LEFTEDGE,
	RIGHTEDGE,
	TOPEDGE,
	BOTTOMEDGE
};

// For panel edge detection on a FOV
struct StPanelEdgeInImage
{
	// Input
	int iLeft;					// ROI
	int iRight;
	int iTop;
	int iBottom;
	PanelEdgeType type;			// type of edge
	int iDecim;					// Decim, valid value = 1, 2, 4;
	double dMinLineLengthRatio;	// Ratio of minimum length of edge
	double dAngleRange;			// Angle range of edge
	
	// Output
	int iFlag;					// rResult flag			
	double dStartRow;				// Line paramter base on original FOV image
	double dSlope;

	StPanelEdgeInImage()
	{
		iDecim= 2;
		dMinLineLengthRatio = 0.5;
		dAngleRange = 3;
		
		Reset();
	}

	void Reset()
	{
		iFlag = 0;
		dStartRow = -1;
		dSlope = 0;
	}
};