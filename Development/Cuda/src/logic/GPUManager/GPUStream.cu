#include "GPUManager.h"
#include "GPUStream.h"
//#include "JobThread.h"

#define FOV_Width 2592
#define FOV_Height 1944

typedef struct {
   float r;
   float i;
} complexf;

namespace CyberGPU
{

	GPUStream::GPUStream(string uniqueName)
	{
		cudaError_t error;

		cudaStreamCreate(&_stream);

		_context = NULL;

		_stdInBuffer.width = _stdOutBuffer.width = FOV_Width;
		_stdInBuffer.height = _stdOutBuffer.height = FOV_Height;
		_stdInBuffer.size = _stdOutBuffer.size = FOV_Width * FOV_Height * sizeof(complexf);

		_stdInBuffer.elements = _stdOutBuffer.elements = NULL;
		error = cudaMalloc((void**)&_stdInBuffer.elements, _stdInBuffer.size);
		error = cudaMalloc((void**)&_stdOutBuffer.elements, _stdOutBuffer.size);
		if (error != cudaSuccess)
		{
			error = cudaSuccess;
		}

		_pGPUJob = NULL;
		_phase = 0;
	}

	GPUStream::~GPUStream(void)
	{
		cudaFree(_stdOutBuffer.elements);
		cudaFree(_stdInBuffer.elements);

		cudaStreamDestroy(_stream);
	}

	void GPUStream::GPUJob(CyberGPU::CGPUJob *pGPUJob)
	{
		_pGPUJob = pGPUJob;
		_phase = 0;
	}
}