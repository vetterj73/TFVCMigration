#pragma once
#include "windows.h"
#include <queue>
#include <string>
using std::queue;
using std::string;
using std::vector;

namespace CyberJob
{
	class GPUJob;
	class GPUJobThread;
	class GPUJobStream;

	typedef void (*CLEAR_JOBSTREAM)(CyberJob::GPUJobStream *jobStream);
	
	///
	///	Interface for a job (a task to perform on a separate thread).
	///
	class GPUJob
	{
	public:
		enum GPUJobStatus
		{
			IDLE,
			ACTIVE,
			WAITING,
			COMPLETED,
		};

		virtual void Run()=0;
		virtual GPUJobStatus GPURun(GPUJobStream *jobStream)=0; // returns job status after function execution

		virtual unsigned int NumberOfStreams()=0;
	};

	///
	///	Sets up a number of job threads that subdivide a big task into little pieces.
	///
	class GPUJobManager
	{
	public:

		///
		///	Constructor.  
		/// Max size of baseName is 32 (truncated to 32 if over 32).
		/// Max number of threads is 99 (changed to 99 if over 99).
		///	
		GPUJobManager(string baseName, unsigned int numThreads, unsigned int numStreams, CLEAR_JOBSTREAM fp);

		///
		///	Destructor
		///
		virtual ~GPUJobManager();

		///
		///	
		/// 
		///
		bool AddAJob(GPUJob* pJob);

		///
		///
		///
		///
		GPUJob* GetNextJob();

		///
		///	MarkAsFinished
		/// Does nothing in the GPUJob implementation 
		///
		void MarkAsFinished() {}

		///
		///	Count of all jobs currently waiting or in process.  When this is 0, we
		/// are done.
		///
		unsigned int TotalJobs();

		DWORD RunGPUThread();

		void LogTimeStamp(std::string msg);
		void DeltaTimeStamp(std::string msg, LARGE_INTEGER starttime);
		void PrintTimeStamps();

	private:

		void ManageStreams();

		bool _killThread;
		HANDLE _GPUThread;
		HANDLE _queueMutex;
		HANDLE _startSignal;
		HANDLE _killSignal;

		HANDLE _logMutex;

		unsigned int _maxStreams;

		vector<GPUJobThread*> _jobThreads;
		vector<GPUJobStream*> _jobStreams;

		queue<GPUJob*>	_jobQueue;

		CLEAR_JOBSTREAM _clearStreamFunctionPointer;

		LARGE_INTEGER _startTime, _frequency;
		queue<std::string> _jobLogs;
	};
};
