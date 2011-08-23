#include "GPUJobManager.h"
#include "GPUJobThread.h"
#include "GPUJobStream.h"
#include "JobThread.h"
#include "JobManager.h"

namespace CyberJob
{
	DWORD WINAPI GPUDeviceThread(LPVOID p)
	{
		GPUJobManager *pGPUJobManager = (GPUJobManager *)p;

		return pGPUJobManager->RunGPUThread();
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
				ManageStreams();
			else
				break;  /// Either there was an issue, or we are done...
		}

		return 0;
	}

	const unsigned int cMaxNameSize = 36;
	GPUJobManager::GPUJobManager(string baseName, unsigned int numThreads, unsigned int numStreams)
	{
		// Validation of lengths...
		string name = baseName;
		if(name.length() > cMaxNameSize-4)
			name = baseName.substr(0, cMaxNameSize-4);
		if(numThreads > 99)
			numThreads = 99;

		for(unsigned int i=0; i<numThreads-1; i++)
		{
			char buf[cMaxNameSize];
			sprintf_s(buf, cMaxNameSize-1, "%s%d", name.c_str(), i);
			string name = buf;
			GPUJobThread* pJT = new GPUJobThread(this, name);
			_jobThreads.push_back(pJT);
		}

		_maxStreams = numStreams;

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

		// Start the GPU thread....
		DWORD d(0);
		_GPUThread = CreateThread(0, 0, GPUDeviceThread, static_cast<LPVOID>(this), 0, &d);

	}

	GPUJobManager::~GPUJobManager()
	{
		for(unsigned int i=0; i<_jobThreads.size(); i++)
		{
			_jobThreads[i]->Kill();
			delete _jobThreads[i];
		}

		for(unsigned int i=0; i<_jobStreams.size(); i++)
		{
			delete _jobStreams[i];
		}

		_jobThreads.clear();
		_jobStreams.clear();

		// Stop the GPUthread.
		SetEvent(_killSignal);

		// Make sure thread stops...
		Sleep(10);

		// Close all handles for cleanup
		CloseHandle(_queueMutex);
		CloseHandle(_startSignal);
		CloseHandle(_killSignal);
		CloseHandle(_GPUThread);

		_GPUThread = NULL;
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
			if (_jobThreads[i]->Status() == GPUJobThread::GPUThreadStatus::ACTIVE) ++count;

		for(unsigned int i=0; i<_jobStreams.size(); i++)
			if (_jobStreams[i]->GPUJob() != NULL) ++count;

		return count;
	}

	void GPUJobManager::ManageStreams()
	{
		bool activeJobs;

		do
		{
			activeJobs = false;
			for (int i=0; i<_maxStreams && i < _jobStreams.size(); ++i)
			{
				if (_jobStreams[i]->GPUJob() == NULL)
				{
					_jobStreams[i]->GPUJob(GetNextJob());
				}

				GPUJob *pGPUJob = _jobStreams[i]->GPUJob();
				if (pGPUJob != NULL)
				{
					activeJobs = true;

					if (pGPUJob->GPURun(_jobStreams[i])) // if job completed
					{
						_jobStreams[i]->GPUJob(NULL);
						--i;
					}
				}
			}
		}
		while (activeJobs);
	}

	bool GPUJobManager::AddAJob(GPUJob* pJob)
	{
		WaitForSingleObject(_queueMutex, INFINITE);
		_jobQueue.push(pJob);
		unsigned int unhandledJobCount = _jobQueue.size();
		ReleaseMutex(_queueMutex);

		unsigned int streamJobCount = 0;
		for(unsigned int i=0; i<_jobStreams.size() && unhandledJobCount > 0; i++)
		{
			if (_jobStreams[i]->GPUJob() != NULL) ++streamJobCount;
		}
		if (streamJobCount < _maxStreams)
		{
			SetEvent(_startSignal);
			if (_maxStreams - streamJobCount >= unhandledJobCount) return true;
		}

		unhandledJobCount -= _maxStreams - streamJobCount;

		for(unsigned int i=0; i<_jobThreads.size() && unhandledJobCount > 0; i++)
		{
			if (_jobThreads[i]->Status() == GPUJobThread::GPUThreadStatus::IDLE)
			{
				_jobThreads[i]->Start();
				--unhandledJobCount;
			}
		}
		return true;
	}
}