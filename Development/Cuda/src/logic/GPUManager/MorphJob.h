#pragma once
#include "GPUManager.h"
#include "UIRect.h"

using namespace CyberGPU;

class Image;
class CyberGPU::GPUStream;

//void ClearMorphJobStream(CyberJob::GPUStream *jobStream);

class MorphJob : public CyberGPU::CGPUJob
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
	CyberGPU::CGPUJob::GPUJobStatus GPURun(CyberGPU::GPUStream *jobStream);

	//unsigned int NumberOfStreams() { return 3; }

	unsigned int OrdinalNumber() { return _ordinal; }

protected:
	CyberGPU::CGPUJob::GPUJobStatus GPUMorphFrom(const Image* pStitched, const Image* pImgIn,
		UIRect roi, CyberGPU::GPUStream *jobStream);

	Image *_pStitched;
	Image *_pFOV;
	UIRect _rect;
	unsigned int _ordinal;
};

CyberGPU::CGPUJob::GPUJobStatus GPUImageMorph(CyberGPU::GPUStream *jobStream, unsigned char* pInBuf,  unsigned int iInSpan, 
	unsigned int iInWidth, unsigned int iInHeight, 
	unsigned char* pOutBuf, unsigned int iOutSpan,
	unsigned int iOutROIStartX, unsigned int iOutROIStartY,
	unsigned int iOutROIWidth, unsigned int iOutROIHeight,
	double dInvTrans[3][3]);
