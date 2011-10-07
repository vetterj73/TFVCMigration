#pragma once
#include "windows.h"
#include <queue>
#include <string>
using std::queue;
using std::string;

#include <cutil.h>
#include <cufft.h>
#include "GPUManager.h"

namespace CyberJob
{

	class GPUJob;

	///
	///	Encapsulates a GPU stream for running jobs.
	///
	class GPUStream
	{
	public:
		///
		///	Constructor - starts the thread
		///	@todo - logging...
		///
		GPUStream(/*GPUManager* pGPUManager, */string uniqueName);
		~GPUStream(void);
 
		cudaStream_t *Stream() { return &_stream; }

		unsigned int Phase() { return _phase; }
		void Phase(unsigned int phase) { _phase = phase; }

		CGPUJob *GPUJob() { return _pGPUJob; }
		void GPUJob(CyberJob::CGPUJob *pGPUJob);

		void *Context() { return _context; }
		void Context(void *context) { _context = context; }

		const ByteMatrix StdInBuffer() { return _stdInBuffer; }
		const ByteMatrix StdOutBuffer() { return _stdOutBuffer; }

		cudaEvent_t *PhaseEvent() { return &_phaseEvent; }

		cufftHandle *Plan() { return &_plan; }

		//GPUJobManager* _pGPUJobManager;

	private:
		///
		///	intentially private default constructor.
		///
		GPUStream(){};
	
		unsigned int _phase;
		cudaStream_t _stream;


		CyberJob::CGPUJob *_pGPUJob;

		///
		///	Job type specific context
		///
		void *_context;

		ByteMatrix _stdInBuffer;
		ByteMatrix _stdOutBuffer;

		cudaEvent_t _phaseEvent;

		cufftHandle _plan;
	};
}