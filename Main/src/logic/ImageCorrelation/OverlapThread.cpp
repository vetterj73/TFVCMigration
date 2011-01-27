#include "OverlapThread.h"
#include "OverlapDefines.h"
#include <string>
using std::string;

DWORD WINAPI ThreadFunc(LPVOID p)
{
	OverlapThread *pOverlapThread = (OverlapThread *)p;
	return pOverlapThread->RunThread();
}

OverlapThread::OverlapThread(int index)
{
	char name[20];
	sprintf_s(name, 20, "QueueMutex%d", index);
	_queueMutex = CreateMutex(0, FALSE, name);

	sprintf_s(name, 20, "QueueSignal%d", index);
	_addSignal = CreateEvent(NULL, false, FALSE, LPCTSTR(name));

	sprintf_s(name, 20, "KillSignal%d", index);
	_killSignal = CreateEvent(NULL, false, FALSE, LPCTSTR(name));

	// Start the thread....
	DWORD d(0);
	_thread = CreateThread(0, 0, ThreadFunc, static_cast<LPVOID>(this), 0, &d);
}

OverlapThread::~OverlapThread(void)
{
	KillThread();
}

void OverlapThread::AddOverlap(Overlap *pOverlap)
{
	WaitForSingleObject(_queueMutex, INFINITE);
	_overlapQueue.push(pOverlap);
	ReleaseMutex(_queueMutex);

	SetEvent(_addSignal);
}

Overlap * OverlapThread::GetNextOverlap()
{
	WaitForSingleObject(_queueMutex, INFINITE);

	if(_overlapQueue.empty())
		return NULL;
	Overlap *pOverlap = _overlapQueue.front();
	_overlapQueue.pop();

	ReleaseMutex(_queueMutex);

	return pOverlap;
}

unsigned int OverlapThread::QueueCount()
{
	unsigned int count=0;
	WaitForSingleObject(_queueMutex, INFINITE);
	count = _overlapQueue.size();
	ReleaseMutex(_queueMutex);
	return count;
}

void OverlapThread::KillThread()
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

DWORD OverlapThread::RunThread()
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

void OverlapThread::ProcessQueue()
{
	Overlap *pOverlap = GetNextOverlap();
	if(pOverlap == NULL)
		return;

	pOverlap->DoIt();
}

