#pragma once
#include "GPUManager.h"
#include "UIRect.h"

using namespace CyberJob;

class Image;
class CyberJob::GPUStream;

//void ClearMorphJobStream(CyberJob::GPUStream *jobStream);

class MorphJob : public CyberJob::CGPUJob
{
public:
	MorphJob(Image* pStitchedImage, Image *pFOV, 
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,
		unsigned int lastRow,
		unsigned int ordinal);
	~MorphJob();

	void Run();
	CyberJob::CGPUJob::GPUJobStatus GPURun(CyberJob::GPUStream *jobStream);

	//unsigned int NumberOfStreams() { return 3; }

	unsigned int OrdinalNumber() { return _ordinal; }

protected:
	Image *_pStitched;
	Image *_pFOV;
	UIRect _rect;
	unsigned int _ordinal;
};

