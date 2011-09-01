#include "GPUJobManager.h"
#include "GPUJobThread.h"
#include "GPUJobStream.h"
//#include "JobThread.h"

namespace CyberJob
{

	GPUJobStream::GPUJobStream(GPUJobManager* pGPUJobManager, string uniqueName)
	{
		_pGPUJobManager = pGPUJobManager;

		cudaStreamCreate(&_stream);

		_context = NULL;

		_pGPUJob = NULL;
		_phase = 0;
	}

	GPUJobStream::~GPUJobStream(void)
	{
		cudaStreamDestroy(_stream);
	}

	void GPUJobStream::GPUJob(CyberJob::GPUJob *pGPUJob)
	{
		_pGPUJob = pGPUJob;
		_phase = 0;
	}
}