#pragma once
#include "windows.h"
#include <queue>
#include <string>
using std::queue;
using std::string;

#include <cutil.h>

namespace CyberJob
{

	class GPUJob;

	///
	///	Encapsulates a GPU stream for running jobs.
	///
	class GPUJobStream
	{
	public:
		///
		///	Constructor - starts the thread
		///	@todo - logging...
		///
		GPUJobStream(GPUJobManager* pGPUJobManager, string uniqueName);
		//GPUJobStream(string uniqueName);
		~GPUJobStream(void);
 
		//void Initialize(GPUJob *pJob);
		///
		///	
		/// 
		///

		///
		///	Stops the thread.  This is called in the destructor
		/// if the client doesn't call it.
		///
		void Kill();

		///
		///	Adds an overlap to the thread queue...
		///
		//void AddJob(GPUJobThread *pOverlap);
		//void Start(/*GPUJob *pJob*/);

		///
		///	Informs client of number of items queued up...
		///
		//unsigned int QueueCount();

		cudaStream_t *Stream() { return &_stream; }

		unsigned int Phase() { return _phase; }
		void Phase(unsigned int phase) { _phase = phase; }

		GPUJob *GPUJob() { return _pGPUJob; }
		void GPUJob(CyberJob::GPUJob *pGPUJob);

		void *Context() { return _context; }
		void Context(void *context) { _context = context; }


	private:
		///
		///	intentially private default constructor.
		///
		GPUJobStream(){};
	
		///
		///	
		///
		GPUJobManager* _pGPUJobManager;

		unsigned int _phase;
		cudaStream_t _stream;
		//cudaEvent_t _phaseEvent;

		HANDLE _statusMutex;

		HANDLE _startSignal;
		HANDLE _killSignal;

		CyberJob::GPUJob *_pGPUJob;
		///
		///	Job type specific context
		///
		void *_context;
	};
}