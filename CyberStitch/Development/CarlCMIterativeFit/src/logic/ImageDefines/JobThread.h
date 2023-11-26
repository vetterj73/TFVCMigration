#pragma once
#include "windows.h"
#include <queue>
#include <string>
using std::queue;
using std::string;

namespace CyberJob
{
	///
	///	Interface for a job (a task to perform on a separate thread).
	///
	class Job
	{
	public:
		virtual void Run()=0;
	};

	///
	///	The purpose of this job is clear the queue for a job thread.  
	/// i.e. - After all legitimate jobs are added, add a LastJob to the queue.
	/// When the queue is empty, you know the thread is not processing anything meaningful.
	///
	class LastJob : public Job
	{
		public:
			void Run(){};
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
		///
		///
		void MarkAsFinished();

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
		///	Basic stuff to ensure thread safety.
		///
		HANDLE _thread;
		HANDLE _queueMutex;
		HANDLE _addSignal;
		HANDLE _killSignal;
		queue<Job*>	_jobQueue;
		LastJob _lastJob;
	};
}