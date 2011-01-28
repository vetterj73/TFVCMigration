#pragma once
#include <string>
#include <vector>
using std::string;
using std::vector;

namespace CyberJob
{
	class JobThread;
	class Job;

	///
	///	Sets up a number of job threads that subdivide a big task into little pieces.
	///
	class JobManager
	{
	public:

		///
		///	Constructor
		///	
		JobManager(string baseName, unsigned int numThreads);

		///
		///	Destructor
		///
		virtual ~JobManager();

		///
		///	Dishes out an equal number of jobs to all threads.
		/// May be enhanced to check which thread(s) are busy...
		///
		void AddAJob(Job* pJob);

		///
		///	Count of all jobs currently waiting or in process.  When this is 0, we
		/// are done.
		///

		unsigned int TotalJobs();

	private:
		vector<JobThread*> _jobThreads;
		int _currentThread;
	};
};