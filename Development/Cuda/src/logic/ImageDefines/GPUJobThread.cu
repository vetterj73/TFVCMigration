#include "GPUJobManager.h"
#include "GPUJobThread.h"
#include "GPUJobStream.h"
#include "JobThread.h"
#include <process.h>
#include "../MosaicDataModel/MorphJob.h"

namespace CyberJob
{

	DWORD WINAPI HostThreadFunc(LPVOID p)
	{
		GPUJobThread *pJobThread = (GPUJobThread *)p;
		return pJobThread->RunHostThread();
	}

	DWORD GPUJobThread::RunHostThread()
	{	
		HANDLE handles[2];
		handles[0] = _startSignal;
		handles[1] = _killSignal;

		while(1)
		{
			DWORD result = WaitForMultipleObjects(2, handles, false, INFINITE);

			if(result == WAIT_OBJECT_0 + 0)
				ProcessQueue();
			else
				break;  /// Either there was an issue, or we are done...
		}

		return 0;
	}

	void GPUJobThread::ProcessQueue()
	{
		_status = GPUJobThread::GPUThreadStatus::ACTIVE;
		
		GPUJob *pGPUJob;

		WaitForSingleObject(_queueMutex, INFINITE);

		while(!_jobQueue.empty())
		{
			pGPUJob = _jobQueue.front();

			ReleaseMutex(_queueMutex);

			char str[128];

			MorphJob* temp = (MorphJob*)pGPUJob;

			sprintf_s(str, "Job %d; Thread %lu;", temp->OrdinalNumber(), GetCurrentThreadId());
			_pGPUJobManager->LogTimeStamp(str);

			pGPUJob->Run();

			sprintf_s(str, "Job %d; Thread %lu; COMPLETE", temp->OrdinalNumber(), GetCurrentThreadId());
			_pGPUJobManager->LogTimeStamp(str);

			WaitForSingleObject(_queueMutex, INFINITE);

			_jobQueue.pop();
			--_totalJobs;
		}

		ReleaseMutex(_queueMutex);
		_status = GPUJobThread::GPUThreadStatus::COMPLETED;
	}
	//DWORD WINAPI GPUJobThread::GPUThreadFunc(LPVOID p)
	//{
	//	GPUJobThread *pGPUJobThread = (GPUJobThread *)p;

	//	char str[128];

	//	MorphJob* temp = (MorphJob*)pGPUJobThread->_pGPUJob;

	//	sprintf_s(str, "Job %d; Thread %lu;", temp->OrdinalNumber(), GetCurrentThreadId());
	//	pGPUJobThread->_pGPUJobManager->LogTimeStamp(str);
	//	pGPUJobThread->_pGPUJob->Run();

	//	pGPUJobThread->_pGPUJob = NULL;
	//	pGPUJobThread->_status = GPUJobThread::GPUThreadStatus::COMPLETED;

	//	sprintf_s(str, "Job %d; Thread %lu; COMPLETE", temp->OrdinalNumber(), GetCurrentThreadId());
	//	pGPUJobThread->_pGPUJobManager->LogTimeStamp(str);
	//	_endthread();
	//	return 0;
	//}

	GPUJobThread::GPUJobThread(GPUJobManager* pGPUJobManager, string uniqueName)
	{
		_pGPUJobManager = pGPUJobManager;
		_status = GPUJobThread::GPUThreadStatus::IDLE;

		_totalJobs = 0;
		_pGPUJob = NULL;
		_threadHandle = NULL;

		string name = uniqueName + "_QueueMutex";
		_queueMutex = CreateMutex(0, FALSE, name.c_str());

		name = uniqueName + "_StartSignal";
		_startSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		name = uniqueName + "_KillSignal";
		_killSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		Status(GPUJobThread::GPUThreadStatus::IDLE);

		// Start the thread....
		DWORD d(0);
		_thread = CreateThread(0, 0, HostThreadFunc, static_cast<LPVOID>(this), 0, &d);
	}

	GPUJobThread::~GPUJobThread(void)
	{
		SetEvent(_killSignal);
	}

	bool GPUJobThread::StartThread()
	{
		GPUJob *pGPUJob = _pGPUJobManager->GetNextJob();

		if (pGPUJob == NULL) return false;

		WaitForSingleObject(_queueMutex, INFINITE);

		_jobQueue.push(pGPUJob);
		++_totalJobs;

		ReleaseMutex(_queueMutex);

		SetEvent(_startSignal);
	}
}