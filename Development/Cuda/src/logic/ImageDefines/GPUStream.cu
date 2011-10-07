#include "GPUManager.h"
#include "GPUStream.h"
//#include "JobThread.h"

#define FOV_Width 2592
#define FOV_Height 1944

namespace CyberJob
{

	GPUStream::GPUStream(/*GPUJobManager* pGPUJobManager, */string uniqueName)
	{
		//_pGPUJobManager = pGPUJobManager;

		cudaStreamCreate(&_stream);

		_context = NULL;

		_stdInBuffer.width = _stdOutBuffer.width = FOV_Width;
		_stdInBuffer.height = _stdOutBuffer.height = FOV_Height;
		_stdInBuffer.size = _stdOutBuffer.size = FOV_Width * FOV_Height * sizeof(unsigned char);

		_stdInBuffer.elements = _stdOutBuffer.elements = NULL;
		cudaMalloc((void**)&_stdInBuffer.elements, _stdInBuffer.size);
		cudaMalloc((void**)&_stdOutBuffer.elements, _stdOutBuffer.size);

		_pGPUJob = NULL;
		_phase = 0;

		_plan = NULL;
	}

	GPUStream::~GPUStream(void)
	{
		cudaFree(_stdOutBuffer.elements);
		cudaFree(_stdInBuffer.elements);

		cudaStreamDestroy(_stream);
	}

	void GPUStream::GPUJob(CyberJob::CGPUJob *pGPUJob)
	{
		_pGPUJob = pGPUJob;
		_phase = 0;
	}
}