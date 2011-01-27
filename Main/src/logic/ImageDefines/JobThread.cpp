#include "JobThread.h"

DWORD WINAPI ThreadFunc(LPVOID p)
{
	JobThread *pJobThread = (JobThread *)p;
	return pJobThread->RunThread();
}

JobThread::JobThread(string uniqueName)
{
	string name = uniqueName + "_QueueMutex";
	_queueMutex = CreateMutex(0, FALSE, name.c_str());

	name = uniqueName + "_QueueSignal";
	_addSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

	name = uniqueName + "_KillSignal";
	_killSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

	// Start the thread....
	DWORD d(0);
	_thread = CreateThread(0, 0, ThreadFunc, static_cast<LPVOID>(this), 0, &d);
}

JobThread::~JobThread(void)
{
	Kill();
}

void JobThread::AddJob(Job *pJob)
{
	WaitForSingleObject(_queueMutex, INFINITE);
	_jobQueue.push(pJob);
	ReleaseMutex(_queueMutex);

	SetEvent(_addSignal);
}

Job * JobThread::GetNextJob()
{
	WaitForSingleObject(_queueMutex, INFINITE);

	Job *pJob = NULL;
	if(!_jobQueue.empty())
	{
		pJob = _jobQueue.front();
		_jobQueue.pop();
	}
	ReleaseMutex(_queueMutex);

	return pJob;
}

unsigned int JobThread::QueueCount()
{
	unsigned int count=0;
	WaitForSingleObject(_queueMutex, INFINITE);
	count = (unsigned int)_jobQueue.size();
	ReleaseMutex(_queueMutex);
	return count;
}

void JobThread::Kill()
{
	if(_thread == NULL)
		return;

	// Stop the thread.
	SetEvent(_killSignal);

	// Make sure thread stops...
	Sleep(10);

	// Close all handles for cleanup
	CloseHandle(_queueMutex);
	CloseHandle(_addSignal);
	CloseHandle(_killSignal);
	CloseHandle(_thread);

	_thread = NULL;
}

DWORD JobThread::RunThread()
{	
	HANDLE handles[2];
	handles[0] = _addSignal;
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

void JobThread::ProcessQueue()
{
	Job *pJob = GetNextJob();

	while(pJob != NULL)
	{
		pJob->Run();
		pJob = GetNextJob();
	}
}

