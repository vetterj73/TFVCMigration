/*-------------------------------------------------------------------------------
         Copyright © 2009 CyberOptics Corporation.  All rights reserved.
---------------------------------------------------------------------------------
    THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
    KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
    PURPOSE.

Module Name:

    SIMCore.cpp

Abstract:

    SIMCore.cpp implements methods for the class SIMCore. This class the static
	access to SIM's Core API.

Environment:

    Unmanaged C++

---------------------------------------------------------------------------------
*/

#include "GPUManager.h"
#include "GPUStream.h"

//#define UNMANAGED_EXPORTS

namespace CyberGPU {

	CGPUJob::CGPUJob()
	{
		_hEvent = CreateEvent(NULL, TRUE, FALSE, NULL);

	}

	CGPUJob::~CGPUJob()
	{
		CloseHandle(_hEvent);
	}

	GPUSTATUS CGPUJob::SetDoneEvent()
	{
		GPUSTATUS status = 0; // !!! status = SUCCESS

		if (!SetEvent(_hEvent))
		{
			DWORD error = GetLastError();
			switch (error)
			{
			default:
				// !!! report error and return status
				status = 1; // !!! status = ERROR
				break;
			}
		}
		return status;
	}

	GPUSTATUS CGPUJob::ResetDoneEvent()
	{
		GPUSTATUS status = 0; // !!! status = SUCCESS

		if (!ResetEvent(_hEvent))
		{
			DWORD error = GetLastError();
			switch (error)
			{
			default:
				// !!! report error and return status
				status = 1; // !!! status = ERROR
				break;
			}
		}
		return status;
	}

	/////////////////////////////////////////////////////////////////////////////
	// SIMAPI construction

	DWORD WINAPI GPUManagerThread(LPVOID p)
	{
		GPUManager *pGPUManager = (GPUManager *)p;

		return pGPUManager->RunGPUThread();
	}

	GPUManager::GPUManager()
	{
	}

	GPUManager::~GPUManager()
	{
		SetEvent(_killSignal);

		_killThread = true;
		while (_killThread) ; // wait for GPU thread to terminate

		//FreeSIMDevices();
	}

	GPUManager* GPUManager::_instance = NULL;
	//int SIMCore::_currentDevice = 0;
	//int SIMCore::_totalDevices = 0;
	//INIT_CALLBACK SIMCore::_registeredInitializationCallback = NULL;
	//void* SIMCore::_registeredInitializationContext = NULL;

	GPUManager& GPUManager::Singleton()
	{
		if( _instance == NULL )
		{
			_instance = new GPUManager;
			GPUSTATUS status = _instance->ConstructGPUManager();
			if (status != 0)
			{
				// !!! report error
			}
		}

		return *_instance;
	}

	GPUSTATUS GPUManager::CheckGPUManager()
	{
		return Singleton().GPUManagerStatus();
	}

	GPUSTATUS GPUManager::RunJob(CGPUJob* pJob, SESSIONHANDLE hSession)
	{
		GPUSTATUS status = RunJobAsynch(pJob, hSession);

		if (status != 0)
			return status;

		WaitForSingleObject(pJob->DoneEvent(), INFINITE);
		// !!! check for DoneEvent, TimeOut or Error and set status

		return status;
	}

	GPUSTATUS GPUManager::RunJobAsynch(CGPUJob* pJob, SESSIONHANDLE hSession)
	{
		GPUSTATUS status = pJob->ResetDoneEvent();

		if (status != 0)
			return status;

		return Singleton().QueueJob(pJob, hSession);
	}

	const unsigned int cMaxNameSize = 36;

	GPUSTATUS GPUManager::GPUManagerStatus( void )
	{
		GPUSTATUS status = 0;
		return status;
	}

	GPUSTATUS GPUManager::ConstructGPUManager( void )
	{
		GPUSTATUS status = 0;

		_maxStreams = 3;

		_killThread = false;

		// initialize SIM devices
		_SessionHandles.clear();
		_SessionContexts.clear();
		_jobStreams.clear();

		_queueMutex = CreateMutex(0, FALSE, "GPUManager_QueueMutex");

		_startSignal = CreateEvent(NULL, FALSE, FALSE, "GPUManager_StartSignal");
		_killSignal = CreateEvent(NULL, FALSE, FALSE, "GPUManager_KillSignal");

		for(unsigned int i=0; i<_maxStreams; i++)
		{
			char buf[cMaxNameSize];
			sprintf_s(buf, cMaxNameSize-1, "GPUManager_Stream_%d", i);
			string name = buf;
			GPUStream* pJS = new GPUStream(/*this, */name);
			_jobStreams.push_back(pJS);
		}


		// Start the GPU thread....
		DWORD d(0);
		_GPUThread = CreateThread(0, 0, GPUManagerThread, static_cast<LPVOID>(this), 0, &d);
		BOOL verify = SetThreadPriority( _GPUThread, THREAD_PRIORITY_ABOVE_NORMAL);

		return status;
	}

	GPUSTATUS GPUManager::QueueJob(CGPUJob* pJob, SESSIONHANDLE hSession)
	{
		GPUSTATUS status = 0; // !!! set to SUCCESS

		WaitForSingleObject(_queueMutex, INFINITE); // !!! search on INFINITE and use timeout
		_jobQueue.push(pJob);
		ReleaseMutex(_queueMutex);

		SetEvent(_startSignal);

		return status;
	}

	CGPUJob* GPUManager::GetNextJob()
	{
		WaitForSingleObject(_queueMutex, INFINITE);

		CGPUJob *pJob = NULL;
		if(!_jobQueue.empty())
		{
			pJob = _jobQueue.front();
			_jobQueue.pop();
		}
		ReleaseMutex(_queueMutex);

		return pJob;
	}

	DWORD GPUManager::RunGPUThread()
	{
		HANDLE handles[2];
		handles[0] = _startSignal;
		handles[1] = _killSignal;

		while(1)
		{
			DWORD result = WaitForMultipleObjects(2, handles, false, INFINITE);

			if(result == WAIT_OBJECT_0 + 0)
			{
				ManageStreams();

				if (_killThread)
				{
					_killThread = false;
					break; // we are done...
				}
				else
				{
					_killThread = true;
				}
			}
			else
				break;  /// Either there was an issue, or we are done...
		}

		return 0;
	}

	void GPUManager::ManageStreams()
	{
		int currentStream = 0;
		bool activeJobs = false;

		do
		{
			for (int i=0; i<_maxStreams && i<_jobStreams.size(); ++i)
			{
				if (_killThread) return;

				if ( _jobStreams[currentStream]->GPUJob() == NULL)
				{
					_jobStreams[currentStream]->GPUJob(GetNextJob());
				}

				CGPUJob *pGPUJob = _jobStreams[currentStream]->GPUJob();
				if (pGPUJob != NULL)
				{
					activeJobs = true;

					CGPUJob::GPUJobStatus status = pGPUJob->GPURun(_jobStreams[currentStream]);

					if (status == CGPUJob::GPUJobStatus::WAITING)
					{
						activeJobs = false;
						break;
					}

					switch (status)
					{
					case CGPUJob::GPUJobStatus::COMPLETED:
						{
							_jobStreams[currentStream]->GPUJob()->SetDoneEvent();
							//char str[128];
							//MorphJob* temp = (MorphJob*)(_jobStreams[currentStream]->GPUJob());

							//sprintf_s(str, "Job %d; Phase %d; COMPLETE", temp->OrdinalNumber(), _jobStreams[currentStream]->Phase());
							//_jobStreams[currentStream]->_pGPUJobManager->LogTimeStamp(str);
						}
						_jobStreams[currentStream]->GPUJob(NULL);
						break;
					case CGPUJob::GPUJobStatus::ACTIVE:
					default:
						break;
					}
				}

				++currentStream;
				if (currentStream >= _maxStreams || currentStream >= _jobStreams.size()) currentStream = 0;
			}
		}
		while (activeJobs);

	}

//	std::string SIMCore::ErrorDescription(SIMSTATUS status)
//	{
//		unsigned long error = (unsigned long)status & 0xFFFFFF;
//		std::string description; 
//		switch(error)
//		{
//		case SIMSTATUS_SUCCESS: description = "completed successfully";
//			break;
//		case SIMSTATUS_UNSUCCESSFUL: description = "completed unsuccessfully";
//			break;
//
//		case SIMSTATUS_USER_ABORT: description = "user abort";
//			break;
//
//		case SIMSTATUS_TIMEOUT: description = "timeout";
//			break;
//		case SIMSTATUS_UNABLE_TO_OPEN_FILE: description = "unable to open file";
//			break;
//		case SIMSTATUS_FILE_NOT_FOUND: description = "file not found";
//			break;
//		case SIMSTATUS_FOLDER_NOT_FOUND: description = "folder not found";
//			break;
//		case SIMSTATUS_FILE_READ_ERROR: description = "file read error";
//			break;
//		case SIMSTATUS_FILE_DATA_CORRUPT: description = "file data corrupt";
//
//// method calling errors
//		case SIMSTATUS_UNRECOGNIZED_COMMAND: description = "unrecognized command";
//			break;
//		case SIMSTATUS_INVALID_COMMAND: description = "invalid command";
//			break;
//		case SIMSTATUS_BAD_ARGUMENT: description = "bad argument";
//			break;
//		case SIMSTATUS_READ_ONLY: description = "read only";
//			break;
//		case SIMSTATUS_UNABLE_TO_LAUNCH_THREAD: description = "unable to launch thread";
//			break;
//
//// device errors
//		case SIMSTATUS_ALREADY_INITIALIZED: description = "already initialized";
//			break;
//		case SIMSTATUS_DEVICE_NOT_READY: description = "device not ready";
//			break;
//		case SIMSTATUS_DEVICE_NOT_FOUND: description = "device not found";
//			break;
//		case SIMSTATUS_UNABLE_TO_OPEN_DEVICE: description = "unable to open device";
//			break;
//		case SIMSTATUS_DEVICE_COMMUNICATION_ERROR: description = "device communication error";
//			break;
//		case SIMSTATUS_SD_CARD_ACCESS_ERROR: description = "SD card access error";
//			break;
//		case SIMSTATUS_COREAPI_UNINITIALIZED: description = "CoreAPI uninitialized";
//			break;
//
//// buffer errors
//		case SIMSTATUS_BUFFERS_UNAVAILABLE: description = "buffers unavailable";
//			break;
//		case SIMSTATUS_ELEMENT_UNAVAILABLE: description = "element unavailable";
//			break;
//		case SIMSTATUS_UNABLE_TO_ALLOCATE: description = "unable to allocate";
//			break;
//		case SIMSTATUS_INCOMPLETE_ALLOCATION: description = "incomplete allocation";
//			break;
//
//// miscellaneous errors
//		case SIMSTATUS_UNABLE_TO_CREATE_EVENT: description = "unable to create event";
//			break;
//		case SIMSTATUS_UNABLE_TO_CREATE_DOM: description = "unable to create DOM";
//			break;
//		case SIMSTATUS_UNABLE_TO_CREATE_THREAD: description = "unable to create thread";
//			break;
//
//// acquisition errors
//		case SIMSTATUS_UNEXPECTED_TRIGGER: description = "unexected trigger";
//			break;
//		case SIMSTATUS_DEVICE_ALREADY_ARMED: description = "device already armed";
//			break;
//		case SIMSTATUS_ACQUISITION_NOT_COMPLETE: description = "aquisition not complete";
//			break;
//		case SIMSTATUS_UNABLE_TO_FIND_FOV_FRAMES: description = "unable to find FOV frames";
//			break;
//
//
//// INITIALIZATION ERRORS
//
//// sensor calibration errors
//		case SIMSTATUS_NO_CALIBRATION_IN_SENSOR: description = "no calibration in SIM sensor";
//			break;
//		case SIMSTATUS_CALIBRATION_SN_MISMATCH: description = "calibration serial number mismatch";
//			break;
//		case SIMSTATUS_CALIBRATION_OUT_OF_DATE: description = "calibration out of date";
//			break;
//
//// more initialization errors
//		case SIMSTATUS_MISSING_DETECTOR: description = "missing detector";
//			break;
//		case SIMSTATUS_DETECTOR_MISMATCH: description = "detector mismatch";
//			break;
//
//		default: description = "unknown SIMStatus";
//			break;
//		}
//		return description;
//	}

}
