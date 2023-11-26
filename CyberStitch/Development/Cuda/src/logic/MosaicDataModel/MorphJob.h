#pragma once
#include "JobThread.h"
#include "UIRect.h"

class Image;
class MorphJob : public CyberJob::Job
{
public:
	MorphJob(Image* pStitchedImage, Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow);

	void Run();

protected:
	Image *_pStitched;
	Image *_pFOV;
	UIRect _rect;
};

