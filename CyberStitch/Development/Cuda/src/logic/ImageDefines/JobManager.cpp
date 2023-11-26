#include "JobManager.h"
#include "JobThread.h"
namespace CyberJob
{
	const unsigned int cMaxNameSize = 36;
	JobManager::JobManager(string baseName, unsigned int numThreads)
	{
		_currentThread = 0;

		//GPUManager::CheckGPUManager();

		// Validation of lengths...
		string name = baseName;
		if(name.length() > cMaxNameSize-4)
			name = baseName.substr(0, cMaxNameSize-4);
		if(numThreads > 99)
			numThreads = 99;

		for(unsigned int i=0; i<numThreads; i++)
		{
			char buf[cMaxNameSize];
			sprintf_s(buf, cMaxNameSize-1, "%s%d", name.c_str(), i);
			string name = buf;
			JobThread* pJT = new JobThread(name);
			_jobThreads.push_back(pJT);
		}
	}

	JobManager::~JobManager()
	{
		for(unsigned int i=0; i<_jobThreads.size(); i++)
		{
			_jobThreads[i]->Kill();
			delete _jobThreads[i];
		}

		_jobThreads.clear();
	}

	///
	///	Dishes out an equal number of jobs to all threads.
	/// May be enhanced to check which thread(s) are busy...
	///
	void JobManager::AddAJob(Job* pJob)
	{
		if(_jobThreads.size() == 0)
		{
			pJob->Run(); // In this case, we are single threaded...
			return;
		}
		_jobThreads[_currentThread]->AddJob(pJob);
		_currentThread++;
		if(_currentThread >= _jobThreads.size())
			_currentThread = 0;
	}	

	void JobManager::MarkAsFinished()
	{
		for(unsigned int i=0; i<_jobThreads.size(); i++)
			_jobThreads[i]->MarkAsFinished();
	}

	unsigned int JobManager::TotalJobs()
	{
		unsigned int count = 0;
		for(unsigned int i=0; i<_jobThreads.size(); i++)
			count += _jobThreads[i]->QueueCount();

		return count;
	}
}