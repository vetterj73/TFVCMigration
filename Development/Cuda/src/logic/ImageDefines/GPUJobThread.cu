#include "GPUJobManager.h"
#include "GPUJobThread.h"
#include "GPUJobStream.h"
#include "JobThread.h"
#include "../MosaicDataModel/MorphJob.h"

namespace CyberJob
{

	DWORD WINAPI GPUJobThread::GPUThreadFunc(LPVOID p)
	{
		GPUJobThread *pGPUJobThread = (GPUJobThread *)p;

		char str[128];

		MorphJob* temp = (MorphJob*)pGPUJobThread->_pGPUJob;

		sprintf_s(str, "Job %d; Thread %lu;", temp->OrdinalNumber(), GetCurrentThreadId());
		pGPUJobThread->_pGPUJobManager->LogTimeStamp(str);
		pGPUJobThread->_pGPUJob->Run();

		pGPUJobThread->_pGPUJob = NULL;
		pGPUJobThread->_status = GPUJobThread::GPUThreadStatus::COMPLETED;

		sprintf_s(str, "Job %d; Thread %lu; COMPLETE", temp->OrdinalNumber(), GetCurrentThreadId());
		pGPUJobThread->_pGPUJobManager->LogTimeStamp(str);
		return 0;
	}

	GPUJobThread::GPUJobThread(GPUJobManager* pGPUJobManager, string uniqueName)
	{
		_pGPUJobManager = pGPUJobManager;
		_status = GPUJobThread::GPUThreadStatus::IDLE;

		_pGPUJob = NULL;
		_threadHandle = NULL;

		//string name = uniqueName + "_StatusMutex";
		//_statusMutex = CreateMutex(0, FALSE, name.c_str());

		//name = uniqueName + "_StartSignal";
		//_startSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		//name = uniqueName + "_KillSignal";
		//_killSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		//Status(GPUJobThread::GPUThreadStatus::IDLE);

		//// Start the thread....
		//DWORD d(0);
		//_thread = CreateThread(0, 0, GPUThreadFunc, static_cast<LPVOID>(this), 0, &d);
	}

	//GPUJobThread::GPUJobThread(GPUJobManager* pGPUJobManager, string uniqueName)
	//{
	//	_pGPUJobManager = pGPUJobManager;

	//	string name = uniqueName + "_StatusMutex";
	//	_statusMutex = CreateMutex(0, FALSE, name.c_str());

	//	name = uniqueName + "_StartSignal";
	//	_startSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

	//	name = uniqueName + "_KillSignal";
	//	_killSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

	//	Status(GPUJobThread::GPUThreadStatus::IDLE);

	//	// Start the thread....
	//	DWORD d(0);
	//	_thread = CreateThread(0, 0, GPUThreadFunc, static_cast<LPVOID>(this), 0, &d);
	//}

	GPUJobThread::~GPUJobThread(void)
	{
		//Kill();
	}

	bool GPUJobThread::LaunchThread()
	{
		_status = GPUJobThread::GPUThreadStatus::ACTIVE;
		_pGPUJob = _pGPUJobManager->GetNextJob();

		if (_pGPUJob != NULL)
		{
			DWORD d(0);
			_threadHandle = CreateThread(0, 0, GPUThreadFunc, static_cast<LPVOID>(this), 0, &d);
			return true;
		}
		
		GPUJobThread::GPUThreadStatus::IDLE;
		return false;
	}

	//void GPUJobThread::Start()
	//{
	//	SetEvent(_startSignal);
	//}

	////void GPUJobThread::MarkAsFinished()
	////{
	////	//AddJob(&_lastJob);
	////}

	//void GPUJobThread::Kill()
	//{
	//	if(_thread == NULL)
	//		return;

	//	// Stop the thread.
	//	SetEvent(_killSignal);

	//	// Make sure thread stops...
	//	Sleep(10);

	//	// Close all handles for cleanup
	//	CloseHandle(_statusMutex);
	//	CloseHandle(_startSignal);
	//	CloseHandle(_killSignal);
	//	CloseHandle(_thread);

	//	_thread = NULL;
	//}

	//DWORD GPUJobThread::RunThread()
	//{	
	//	HANDLE handles[2];
	//	handles[0] = _startSignal;
	//	handles[1] = _killSignal;

	//	while(1)
	//	{
	//		DWORD result = WaitForMultipleObjects(2, handles, false, INFINITE);

	//		if(result == WAIT_OBJECT_0 + 0)
	//			ProcessQueue();
	//		else
	//			break;  /// Either there was an issue, or we are done...
	//	}

	//	return 0;
	//}

	//void GPUJobThread::ProcessQueue()
	//{
	//	GPUJob *pJob = _pGPUJobManager->GetNextJob();

	//	while(pJob != NULL)
	//	{
	//		Status(GPUJobThread::GPUThreadStatus::ACTIVE);


	//		char str[128];

	//		MorphJob* temp = (MorphJob*)pJob;

	//		sprintf_s(str, "Job %d; Thread %lu;", temp->OrdinalNumber(), GetCurrentThreadId());
	//		_pGPUJobManager->LogTimeStamp(str);
	//		pJob->Run();
	//		pJob = _pGPUJobManager->GetNextJob();
	//	}

	//	Status(GPUJobThread::GPUThreadStatus::IDLE);
	//}
}