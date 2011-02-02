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
		///	Constructor.  
		/// Max size of baseName is 32 (truncated to 32 if over 32).
		/// Max number of threads is 99 (changed to 99 if over 99).
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
		///	MarkAsFinished
		/// This lets the manager know that you are done.  It does not reset anything, it just 
		/// adds a "complete" event to each thread so that the manager can accurately track
		/// the progress of real jobs through the system.  User Beware, this does not stop you from
		/// adding legitimate jobs after you specify that you are done!
		///
		void MarkAsFinished();

		///
		///	Count of all jobs currently waiting or in process.  When this is 0, we
		/// are done.
		///

		unsigned int TotalJobs();

	private:
		vector<JobThread*> _jobThreads;
		unsigned int _currentThread;
	};
};