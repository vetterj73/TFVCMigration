#pragma once
#include "windows.h"
#include <queue>
#include <string>
#include <vector>
using std::queue;
using std::string;
using std::vector;

///
///	Interface for a job (a task to perform on a separate thread).
///
class Job
{
public:
	virtual void Run()=0;
};

///
///	Encapsulates a Thread for running jobs.
///
class JobThread
{
public:
	///
	///	Constructor - starts the thread
	///	@todo - logging...
	///
	JobThread(string uniqueName);
	~JobThread(void);
 
	///
	///	Stops the thread.  This is called in the destructor
	/// if the client doesn't call it.
	///
	void Kill();

	///
	///	Adds an overlap to the thread queue...
	///
	void AddJob(Job *pOverlap);

	///
	///	Informs client of number of items queued up...
	///
	unsigned int QueueCount();

	///
	///	This should not be called by client.
	///
	DWORD RunThread();

private:
	///
	///	intentially private default constructor.
	///
	JobThread(){};
	
	void ProcessQueue();
	Job * GetNextJob();

	///
	///	Basic stuff to ensure
	///
	HANDLE _thread;
	HANDLE _queueMutex;
	HANDLE _addSignal;
	HANDLE _killSignal;
	queue<Job*>	_jobQueue;
};

///
///	Sets up a number of job threads to perform a task.
///
class JobManager
{
public:

	///
	///	Constructor
	///	
	JobManager(string baseName, unsigned int numThreads)
	{
		_currentThread = 0;
		for(unsigned int i=0; i<numThreads; i++)
		{
			char buf[20];
			sprintf_s(buf, 19, "%s%d", baseName.c_str(), i);
			string name = buf;
			JobThread* pJT = new JobThread(name);
			_jobThreads.push_back(pJT);
		}
	}

	virtual ~JobManager()
	{
		for(int i=0; i<_jobThreads.size(); i++)
		{
			_jobThreads[i]->Kill();
			delete _jobThreads[i];
		}

		_jobThreads.clear();
	}

	///
	///	Dishes out an equal number of jobs to all threads.
	/// May be enhanced to check which thread(s) are busy...
	///
	void AddJob(Job* pJob)
	{
		if(_jobThreads.size() == 0)
		{
			pJob->Run(); // In this case, we are single threaded...
			return;
		}
		_jobThreads[_currentThread]->AddJob(pJob);
		_currentThread++;
		if(_currentThread >= _jobThreads.size())
			_currentThread = 0;
	}	

	///
	///	Count of all jobs currently waiting or in process.  When this is 0, we
	/// are done.
	///
	unsigned int TotalJobs()
	{
		unsigned int count = 0;
		for(unsigned int i=0; i<_jobThreads.size(); i++)
			count += _jobThreads[i]->QueueCount();

		return count;
	}

private:
	vector<JobThread*> _jobThreads;
	int _currentThread;
};

