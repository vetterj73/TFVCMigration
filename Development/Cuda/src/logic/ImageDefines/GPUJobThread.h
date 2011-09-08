#pragma once
#include "windows.h"
#include <queue>
#include <string>

using std::queue;
using std::string;

namespace CyberJob
{
	// For now, the number of streams = the number of phases.
	// after each phase set an event for each stream.
	// after the second phase wait on the event created for the first stream in the preceeding phase
	// after the next phase wait on the event created in the first phase for the next stream
	// after the last phase clear the job and add a new job

	///
	///	Encapsulates a Thread for running jobs.
	///
	class GPUJobThread
	{
	public:

		enum GPUThreadStatus
		{
			IDLE,
			ACTIVE,
			COMPLETED,
		};

		///
		///	Constructor
		///
		GPUJobThread::GPUJobThread(GPUJobManager* pGPUJobManager, string uniqueName);
		//GPUJobThread(string uniqueName);
		~GPUJobThread(void);
 
		///
		///	
		/// 
		GPUThreadStatus Status() { return _status; }
		void Status(GPUThreadStatus status) { _status = status; }

		///
		///
		///
		unsigned int TotalJobs() { return _totalJobs; }

		///
		///
		///
		bool StartThread();

		DWORD RunHostThread();

	private:

		void ProcessQueue();

		GPUJobManager* _pGPUJobManager;

		GPUThreadStatus _status;
		GPUJob* _pGPUJob;
		HANDLE _threadHandle;

		unsigned int _totalJobs;

		HANDLE _queueMutex;
		HANDLE _startSignal;
		HANDLE _killSignal;
		queue<GPUJob*>	_jobQueue;
		HANDLE _thread;
	};
}