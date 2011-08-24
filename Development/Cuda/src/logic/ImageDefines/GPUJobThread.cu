#include "GPUJobManager.h"
#include "GPUJobThread.h"
#include "GPUJobStream.h"
#include "JobThread.h"

namespace CyberJob
{

	DWORD WINAPI GPUThreadFunc(LPVOID p)
	{
		GPUJobThread *pGPUJobThread = (GPUJobThread *)p;
		return pGPUJobThread->RunThread();
	}

	GPUJobThread::GPUJobThread(GPUJobManager* pGPUJobManager, string uniqueName)
	{
		_pGPUJobManager = pGPUJobManager;

		string name = uniqueName + "_StatusMutex";
		_statusMutex = CreateMutex(0, FALSE, name.c_str());

		name = uniqueName + "_StartSignal";
		_startSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		name = uniqueName + "_KillSignal";
		_killSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		Status(GPUJobThread::GPUThreadStatus::IDLE);

		// Start the thread....
		DWORD d(0);
		_thread = CreateThread(0, 0, GPUThreadFunc, static_cast<LPVOID>(this), 0, &d);
	}

	GPUJobThread::~GPUJobThread(void)
	{
		Kill();
	}

	void GPUJobThread::Start()
	{
		SetEvent(_startSignal);
	}

	GPUJobThread::GPUThreadStatus GPUJobThread::Status()
	{
		GPUThreadStatus status;

		WaitForSingleObject(_statusMutex, INFINITE);
		status = _status;
		ReleaseMutex(_statusMutex);
		return status;
	}

	void GPUJobThread::Status(GPUThreadStatus status)
	{
		WaitForSingleObject(_statusMutex, INFINITE);
		_status = status;
		ReleaseMutex(_statusMutex);
	}

	//void GPUJobThread::MarkAsFinished()
	//{
	//	//AddJob(&_lastJob);
	//}

	void GPUJobThread::Kill()
	{
		if(_thread == NULL)
			return;

		// Stop the thread.
		SetEvent(_killSignal);

		// Make sure thread stops...
		Sleep(10);

		// Close all handles for cleanup
		CloseHandle(_statusMutex);
		CloseHandle(_startSignal);
		CloseHandle(_killSignal);
		CloseHandle(_thread);

		_thread = NULL;
	}

	DWORD GPUJobThread::RunThread()
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
		GPUJob *pJob = _pGPUJobManager->GetNextJob();

		while(pJob != NULL)
		{
			Status(GPUJobThread::GPUThreadStatus::ACTIVE);

			pJob->Run();
			pJob = _pGPUJobManager->GetNextJob();
		}

		Status(GPUJobThread::GPUThreadStatus::IDLE);
	}
}