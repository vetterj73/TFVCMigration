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

//	class JobStream
//	{
//	public:
//	};
//
//	class CGPUThreadData
//	{
//	//#pragma region Friends
//
//	//	friend CSIMHdwDevice;
//
//	//#pragma endregion
//
//	//#pragma region Enumerations
//
//	//	enum FrameXferStatus
//	//	{
//	//		Disabled,
//	//		Inactive,
//	//		Xfering,
//	//		Processing,
//	//		Finished
//	//	};
//
//	//#pragma endregion
//
//	protected:
//
////		int _row;
////		int _camera;
////		int _triggerlist;
////		TriggerListType _TLtype;
////
////		FrameXferStatus _status;
////		DWORD _pTargetBuffer;
//////	public:
////		ProcessFrameThreadData _frameThreadData;
//		HANDLE _threadID;
//	};

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
		};

		///
		///	Constructor - starts the thread
		///	@todo - logging...
		///
		GPUJobThread::GPUJobThread(GPUJobManager* pGPUJobManager, string uniqueName);
		//GPUJobThread(string uniqueName);
		~GPUJobThread(void);
 
		///
		///	
		/// 
		///
		GPUThreadStatus Status();
		void Status(GPUThreadStatus status);

		///
		///
		///
		void Start();

		///
		///	Stops the thread.  This is called in the destructor
		/// if the client doesn't call it.
		///
		void Kill();

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
		GPUJobThread(){};
	
		void ProcessQueue();
		GPUJob * GetNextJob();

		///
		///	Basic stuff to ensure thread safety.
		///
		GPUJobManager* _pGPUJobManager;

		GPUThreadStatus _status;

		HANDLE _thread;

		HANDLE _statusMutex;

		HANDLE _startSignal;
		HANDLE _killSignal;
		//GPUJob* _startJob;
	};
}