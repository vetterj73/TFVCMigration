#pragma once
#include "GPUJobManager.h"
#include "UIRect.h"

using namespace CyberJob;

class Image;
class CyberJob::GPUJobStream;

class MorphJob : public CyberJob::GPUJob
{
public:
	MorphJob(Image* pStitchedImage, Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow,
		unsigned int ordinal);

	void Run();
	bool GPURun(CyberJob::GPUJobStream *jobStream); // true = job done, false = more to do

	unsigned int NumberOfStreams() { return 3; }

protected:
	Image *_pStitched;
	Image *_pFOV;
	UIRect _rect;
	unsigned int _ordinal;
};

