#include "JobManager.h"
#include "JobThread.h"
namespace CyberJob
{

	JobManager::JobManager(string baseName, unsigned int numThreads)
	{
		_currentThread = 0;
		for(unsigned int i=0; i<numThreads; i++)
		{
			char buf[20];
			sprintf_s(buf, 19, "%s%d", baseName.c_str(), i);
			string name = buf;
			JobThread* pJT = new JobThread(name);
			_jobThreads.push_back(pJT);
		}
	}

	JobManager::~JobManager()
	{
		for(int i=0; i<_jobThreads.size(); i++)
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

	unsigned int JobManager::TotalJobs()
	{
		unsigned int count = 0;
		for(unsigned int i=0; i<_jobThreads.size(); i++)
			count += _jobThreads[i]->QueueCount();

		return count;
	}
}