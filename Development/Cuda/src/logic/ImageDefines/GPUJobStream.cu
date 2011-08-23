#include "GPUJobManager.h"
#include "GPUJobThread.h"
#include "GPUJobStream.h"
//#include "JobThread.h"

namespace CyberJob
{

	GPUJobStream::GPUJobStream(GPUJobManager* pGPUJobManager, string uniqueName)
	{
		_pGPUJobManager = pGPUJobManager;

		string name = uniqueName + "_StatusMutex";
		_statusMutex = CreateMutex(0, FALSE, name.c_str());

		cudaStreamCreate(&_stream);

		cudaEventCreate(&_phaseEvent);
	}

	GPUJobStream::~GPUJobStream(void)
	{
		//Kill();

		//// Make sure thread stops...
		//Sleep(10);

		cudaEventDestroy(_phaseEvent);

		cudaStreamDestroy(_stream);
	}

	void GPUJobStream::Initialize(GPUJob *pJob)
	{
	}

	void GPUJobStream::Kill()
	{
		//if(_thread == NULL)
		//	return;

		//// Stop the thread.
		//SetEvent(_killSignal);

		//// Make sure thread stops...
		//Sleep(10);

		//// Close all handles for cleanup
		//CloseHandle(_queueMutex);
		//CloseHandle(_addSignal);
		//CloseHandle(_killSignal);
		//CloseHandle(_thread);

		//_thread = NULL;
	}
}