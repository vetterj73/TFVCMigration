#pragma once
#include "GPUJobManager.h"
#include "UIRect.h"

using namespace CyberJob;

class Image;
class CyberJob::GPUJobStream;

void ClearMorphJobStream(CyberJob::GPUJobStream *jobStream);

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
	CyberJob::GPUJob::GPUJobStatus GPURun(CyberJob::GPUJobStream *jobStream);

	unsigned int NumberOfStreams() { return 3; }

	unsigned int OrdinalNumber() { return _ordinal; }

protected:
	Image *_pStitched;
	Image *_pFOV;
	UIRect _rect;
	unsigned int _ordinal;
};

