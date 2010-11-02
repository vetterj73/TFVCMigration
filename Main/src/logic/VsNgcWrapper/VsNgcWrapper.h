#pragma once

#include <windows.h>

class VsNgcAlignment;

// Input for NGC alignment
struct NgcParams
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
struct NgcResults
{
		// Outputs
	double dMatchPosX;				// The Matching location in target image after search
	double dMatchPosY;
	double dCoreScore;
	double dAmbigScore;

	NgcResults()
	{
		dMatchPosX = 0;
		dMatchPosY = 0;
		dCoreScore = 0;
		dAmbigScore = -1;
	};
};


class VsNgcWrapper
{
public:
	VsNgcWrapper();
	~VsNgcWrapper(void);

	bool Align(NgcParams params, NgcResults* pResults);

private:

	VsNgcAlignment* _pAlignNgc;
};

