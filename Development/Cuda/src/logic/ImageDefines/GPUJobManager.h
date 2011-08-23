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


	///
	///	Interface for a job (a task to perform on a separate thread).
	///
	class GPUJob
	{
	public:
		virtual void Run()=0;
		virtual bool GPURun(GPUJobStream *jobStream)=0; // true = job done, false = more to do

		//virtual unsigned int NumberOfPhases()=0;
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
		GPUJobManager(string baseName, unsigned int numThreads, unsigned int numStreams);

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


	private:

		void ManageStreams();
		void SetupGPUStreams(unsigned int count);

		//std::string _name;

		HANDLE _GPUThread;
		HANDLE _queueMutex;
		HANDLE _streamMutex;
		HANDLE _startSignal;
		HANDLE _killSignal;

		vector<GPUJobThread*> _jobThreads;
		vector<GPUJobStream*> _jobStreams;

		unsigned int _maxStreams;
		vector<GPUJob*> _GPUStreamQueue;

		queue<GPUJob*>	_jobQueue;
		//unsigned int _currentJob;
	};
};

//#include "GPUJobThread.h"
//#include "GPUJobStream.h"
