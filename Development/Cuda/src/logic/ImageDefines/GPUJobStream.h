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
		~GPUJobStream(void);
 
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
	
		GPUJobManager* _pGPUJobManager;

		unsigned int _phase;
		cudaStream_t _stream;


		CyberJob::GPUJob *_pGPUJob;

		///
		///	Job type specific context
		///
		void *_context;
	};
}