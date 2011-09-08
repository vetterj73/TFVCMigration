#include "GPUJobManager.h"
#include "GPUJobThread.h"
#include "GPUJobStream.h"
#include "JobThread.h"
#include "JobManager.h"
#include "../MosaicDataModel/MorphJob.h"
#include <assert.h>

namespace CyberJob
{
	DWORD WINAPI GPUMainThread(LPVOID p)
	{
		GPUJobManager *pGPUJobManager = (GPUJobManager *)p;

		return pGPUJobManager->RunGPUThread();
	}

	const unsigned int cMaxNameSize = 36;
	GPUJobManager::GPUJobManager(string baseName, unsigned int numThreads, unsigned int numStreams, CLEAR_JOBSTREAM fp)
	{
		_killThread = false;
		_clearStreamFunctionPointer = fp;

		_maxStreams = numStreams;
		int maxThreads = numThreads;

		// Validation of string lengths...
		string name = baseName;
		if(name.length() > cMaxNameSize-4)
			name = baseName.substr(0, cMaxNameSize-4);
		if(numThreads > 99)
			numThreads = 99;

		for(int i=0; i<maxThreads-1; i++)
		{
			char buf[cMaxNameSize];
			sprintf_s(buf, cMaxNameSize-1, "%s%d", name.c_str(), i);
			string name = buf;
			GPUJobThread* pJT = new GPUJobThread(this, name);
			_jobThreads.push_back(pJT);
		}

		for(unsigned int i=0; i<_maxStreams; i++)
		{
			char buf[cMaxNameSize];
			sprintf_s(buf, cMaxNameSize-1, "%s%d", name.c_str(), i);
			string name = buf;
			GPUJobStream* pJS = new GPUJobStream(this, name);
			_jobStreams.push_back(pJS);
		}

		name = "GPUJobManager_QueueMutex";
		_queueMutex = CreateMutex(0, FALSE, name.c_str());

		name = "GPUJobManager_StartSignal";
		_startSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		name = "GPUJobManager_KillSignal";
		_killSignal = CreateEvent(NULL, FALSE, FALSE, name.c_str());

		name = "GPUJobManager_LogMutex";
		_logMutex = CreateMutex(0, FALSE, name.c_str());
		assert(::QueryPerformanceCounter(&_startTime));
		assert(::QueryPerformanceFrequency(&_frequency));


		// Start the GPU thread....
		DWORD d(0);
		_GPUThread = CreateThread(0, 0, GPUMainThread, static_cast<LPVOID>(this), 0, &d);
		BOOL verify = SetThreadPriority( _GPUThread, THREAD_PRIORITY_ABOVE_NORMAL);
	}

	GPUJobManager::~GPUJobManager()
	{
		char str[128];
		sprintf_s(str, "MorphJob Done!!");
		
		LogTimeStamp(str);
		PrintTimeStamps();

		SetEvent(_killSignal);

		_killThread = true;
		while (_killThread) ; // wait for GPU thread to terminate

		for(unsigned int i=0; i<_jobThreads.size(); i++)
		{
			delete _jobThreads[i];
		}

		for(unsigned int i=0; i<_jobStreams.size(); i++)
		{
			if (_clearStreamFunctionPointer != NULL)
				_clearStreamFunctionPointer(_jobStreams[i]);
			delete _jobStreams[i];
		}

		_jobThreads.clear();
		_jobStreams.clear();

		// Stop the GPUthread.
		// Make sure thread stops...
		Sleep(10);

		// Close all handles for cleanup
		CloseHandle(_queueMutex);
		CloseHandle(_startSignal);
		CloseHandle(_killSignal);
		CloseHandle(_GPUThread);

		CloseHandle(_logMutex);

		_GPUThread = NULL;
}

	bool GPUJobManager::AddAJob(GPUJob* pJob)
	{
		WaitForSingleObject(_queueMutex, INFINITE);
		_jobQueue.push(pJob);
		unsigned int unhandledJobCount = _jobQueue.size();
		ReleaseMutex(_queueMutex);

		SetEvent(_startSignal);

		return true;
	}

	GPUJob* GPUJobManager::GetNextJob()
	{
		WaitForSingleObject(_queueMutex, INFINITE);

		GPUJob *pJob = NULL;
		if(!_jobQueue.empty())
		{
			pJob = _jobQueue.front();
			_jobQueue.pop();
		}
		ReleaseMutex(_queueMutex);

		return pJob;
	}

	unsigned int GPUJobManager::TotalJobs()
	{

		unsigned int count = 0;

		WaitForSingleObject(_queueMutex, INFINITE);
		count += _jobQueue.size();
		ReleaseMutex(_queueMutex);

		for(unsigned int i=0; i<_jobThreads.size(); i++)
			count += _jobThreads[i]->TotalJobs();

		for(unsigned int i=0; i<_jobStreams.size(); i++)
			if (_jobStreams[i]->GPUJob() != NULL) ++count;

		if (count <= 0)
			count = _jobQueue.size();

		return count;
	}

	DWORD GPUJobManager::RunGPUThread()
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

	void GPUJobManager::ManageStreams()
	{
		int currentStream = 0;
		bool activeJobs = false;
		bool currentJobs = false;

		do
		{
			do
			{
				for (int i=0; i<_maxStreams && i<_jobStreams.size(); ++i)
				{
					if (_killThread) return;

					if ( _jobStreams[currentStream]->GPUJob() == NULL)
					{
						_jobStreams[currentStream]->GPUJob(GetNextJob());
					}

					GPUJob *pGPUJob = _jobStreams[currentStream]->GPUJob();
					if (pGPUJob != NULL)
					{
						activeJobs = currentJobs = true;

						GPUJob::GPUJobStatus status = pGPUJob->GPURun(_jobStreams[currentStream]);

						if (status == GPUJob::GPUJobStatus::WAITING)
						{
							activeJobs = false;
							break;
						}

						switch (status)
						{
						case GPUJob::GPUJobStatus::COMPLETED:
							{
								char str[128];
								MorphJob* temp = (MorphJob*)(_jobStreams[currentStream]->GPUJob());

								sprintf_s(str, "Job %d; Phase %d; COMPLETE", temp->OrdinalNumber(), _jobStreams[currentStream]->Phase());
								_jobStreams[currentStream]->_pGPUJobManager->LogTimeStamp(str);
							}
							_jobStreams[currentStream]->GPUJob(NULL);
							break;
						case GPUJob::GPUJobStatus::ACTIVE:
						default:
							break;
						}
					}

					++currentStream;
					if (currentStream >= _maxStreams || currentStream >= _jobStreams.size()) currentStream = 0;
				}
			}
			while (activeJobs);

			if (_jobQueue.size() > 20 || _maxStreams == 0)
			{
				for (int i=0; i<_jobThreads.size(); ++i)
				{
					if (_killThread) return;

					if (_jobThreads[i]->TotalJobs() == 0 || (_maxStreams == 0 && _jobThreads[i]->TotalJobs() < 2))
					{
						if (_jobThreads[i]->StartThread())
						{
							currentJobs = true;
						}
					}
				}
			}
			if (_maxStreams == 0) Sleep(1);
		}

		while (currentJobs);
	}

	void GPUJobManager::PrintTimeStamps()
	{
		WaitForSingleObject(_logMutex, INFINITE);
		while (_jobLogs.size() > 0)
		{
			std::string str = _jobLogs.front();
			printf_s(str.c_str());
			_jobLogs.pop();
		}
		ReleaseMutex(_logMutex);
	}

	void GPUJobManager::DeltaTimeStamp(std::string msg, LARGE_INTEGER starttime)
	{
		char str[128];

		LARGE_INTEGER timestamp;
		assert(::QueryPerformanceCounter(&timestamp));
		LARGE_INTEGER deltaTime;
		deltaTime.QuadPart = timestamp.QuadPart - starttime.QuadPart;

		sprintf_s(str, 127, "<%.6f>%s\n",
			(static_cast<double>(deltaTime.QuadPart) / static_cast<double>(_frequency.QuadPart)), msg.c_str());

		WaitForSingleObject(_logMutex, INFINITE);
		_jobLogs.push(str);
		ReleaseMutex(_logMutex);
	}
	void GPUJobManager::LogTimeStamp(std::string msg)
	{
		char str[128];

		LARGE_INTEGER timestamp;
		assert(::QueryPerformanceCounter(&timestamp));
		LARGE_INTEGER deltaTime;
		deltaTime.QuadPart = timestamp.QuadPart - _startTime.QuadPart;

		sprintf_s(str, 127, "<%.6f>%s\n",
			(static_cast<double>(deltaTime.QuadPart) / static_cast<double>(_frequency.QuadPart)), msg.c_str());

		WaitForSingleObject(_logMutex, INFINITE);
		_jobLogs.push(str);
		ReleaseMutex(_logMutex);
	}
}