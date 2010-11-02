#pragma once

#include "VsEnvironManager.h"

// Input for NGC alignment
struct NgcAlignParams
{
	unsigned char* pcTemplateBuf;
	unsigned int   iTemplateImWidth;
	unsigned int   iTemplateImHeight;	
	unsigned int   iTemplateImSpan;
	unsigned int   iTemplateLeft;
	unsigned int   iTemplateRight;	
	unsigned int   iTemplateTop;
	unsigned int   iTemplateBottom;

	unsigned char* pcSearchBuf;
	unsigned int   iSearchImWidth;
	unsigned int   iSearchImHeight;	
	unsigned int   iSearchImSpan;
	unsigned int   iSearchLeft;
	unsigned int   iSearchRight;	
	unsigned int   iSearchTop;
	unsigned int   iSearchBottom;

	bool		   bUseMask;
	unsigned char* pcMaskBuf;
};

// Output for NGC alignment
struct NgcAlignResults
{
		// Outputs
	double dMatchPosX;				// The Matching location in target image after search
	double dMatchPosY;
	double dCoreScore;
	double dAmbigScore;

	NgcAlignResults()
	{
		dMatchPosX = 0;
		dMatchPosY = 0;
		dCoreScore = 0;
		dAmbigScore = -1;
	};
};

// NGC errors
enum PatchAlignResult {
	Found			= 0,
	UniformTemplate = 1,
	OtherTempFailure= 2,
	CreateCorFailure= 3,
	NoMatch			= 4,
	FlatPeak		= 5,
	TimeOut			= 6,
	UnknownFailure	= 7,
	MaskFailure		= 8,
	NotRun			= 9
};

// Struct for image patch alignment
struct ImPatchAlignStruct
{
	// Inputs
	VsCamImage oImTemplate;			// The image that template is gotten from 
	VsCamImage oImSearch;			// The image used for search
	VsStToolRect tRectTemplate;		// The rectangle for template in source image
	VsStToolRect tRectSearch;		// The rectangle for seach area in target image

	bool bMask;
	VsCamImage oMaskIm;				// The mask image
	
	// Vs parameters	
	int iDepth;						// Search depth
	int iMinTempStdDev;				// Any template with STD<iMinTempStdDev will be considered uniform template
	float fCorHiResMinScore;		// Any location with correlation score<fCorHiResMinScore in highest resoltuion will not be counted
	float fCorGainTolerance;		// Any location with Gain>abs(1-fCorGainTolerance) will not be count
	float fCorLoResMinScore;		// Any location with correlation score<fCorLoResMinScore in lowest resoltuion will not be counted

	// Outputs
	double dMatchPosX;				// The Matching location in target image after search
	double dMatchPosY;
	double dCoreScore;
	double dAmbigScore;

	PatchAlignResult eResult;		// Alignment results

	ImPatchAlignStruct()
	{
		eResult = NotRun;

		iMinTempStdDev		= 8;
		fCorGainTolerance	= 0.3f;
		fCorLoResMinScore	= 0.4f;
		fCorHiResMinScore	= 0.5f;
	}
};

class VsNgcAlignment
{
public:
	VsNgcAlignment(NgcAlignParams params);
	~VsNgcAlignment(void);

	bool Align(NgcAlignResults* pResult);

private:
	bool Align();

	VsEnviron _oVsEnv;
	ImPatchAlignStruct _alignSt;
};

