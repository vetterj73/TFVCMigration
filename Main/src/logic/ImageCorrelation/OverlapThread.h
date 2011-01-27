#pragma once
#include "windows.h"
#include <queue>
using std::queue;

class Overlap;

///
///	This class is used to threading overlap regions.
///
class OverlapThread
{
public:
	///
	///	Constructor - starts the thread
	///	@todo - logging...
	///
	OverlapThread(int index);
	~OverlapThread(void);

	///
	///	Adds an overlap to the thread queue...
	///
	void AddOverlap(Overlap *pOverlap);

	///
	///	Informs client of number of items queued up...
	///
	unsigned int QueueCount();

	///
	///	Kills the thread when ready.  This is called in the destructor
	/// if the client doesn't call it.
	///
	void KillThread();
	
	///
	///	This should not be called by client.
	///
	DWORD RunThread();

private:
	///
	///	intentially private default constructor.
	///
	OverlapThread(){};
	
	void ProcessQueue();
	Overlap * GetNextOverlap();

	///
	///	Basic stuff to ensure
	///
	HANDLE _thread;
	HANDLE _queueMutex;
	HANDLE _addSignal;
	HANDLE _killSignal;
	queue<Overlap*>	_overlapQueue;
};

