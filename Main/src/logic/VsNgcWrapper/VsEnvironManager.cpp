#include "VsEnvironManager.h"

#include <map>
using std::map;

//Singleton pattern
VsEnvironManager& VsEnvironManager::Instance(void)
{
	static VsEnvironManager inst;
	return inst;
}

VsEnvironManager::VsEnvironManager(void)
{
}

// Pass environment point in before an inside environment is created
// return ture if success
//		false if an inside environment is already created
bool VsEnvironManager::SetEnv(VsEnviron& env, DWORD threadId)
{
	if(GetStaticEnv()) return(false);

	GetStaticEnv() = env;
	GetEnvThread() = threadId;
	
	return(true);
}

VsEnvironManager::~VsEnvironManager(void)
{
	DisposeEnv();
}

VsEnviron& VsEnvironManager::GetStaticEnv()
{
	static VsEnviron retVal = 0;
	return retVal;
}

DWORD& VsEnvironManager::GetEnvThread()
{
	static DWORD retVal = 0;
	return retVal;
}

int VsEnvironManager::GetAddEnvUseCount( int add/*=0*/ )
{
	::CRITICAL_SECTION cs;
	::InitializeCriticalSection(&cs);
	::EnterCriticalSection(&cs);

	// int with a default initialization to 0
	struct UseNum { UseNum() : num(0) {} int num; }; 
	static std::map<DWORD,UseNum> useCount;

	useCount[::GetCurrentThreadId()].num += add;
	
	int retVal = useCount[::GetCurrentThreadId()].num;
	
	::LeaveCriticalSection(&cs);
	::DeleteCriticalSection(&cs);
	return retVal;
}


VsEnviron VsEnvironManager::GetEnv()
{
	VsEnviron& retVal = GetStaticEnv();

	if (!retVal)
	{
		retVal = ::vsCreateSharedVisionEnviron(0, 0);
		GetEnvThread() = ::GetCurrentThreadId();
	}
	
	DWORD envThread = GetEnvThread();

	if (envThread != ::GetCurrentThreadId() )
	{
		int useCount = GetAddEnvUseCount(+1);
		if (useCount == 1)
		{
			if(vsAttachThread() == -1)
				throw std::runtime_error("vsAttachThread() failed");
		}
	}
	return retVal;
}


void VsEnvironManager::ReleaseEnv()
{
	DWORD envThread = GetEnvThread();

	if (envThread != ::GetCurrentThreadId() )
	{
		int useCount = GetAddEnvUseCount(-1);
		if( useCount == 0 )
		{
			if (vsDetachThread(NULL) == -1)
				throw std::runtime_error("vsDetachThread() failed");
		}
		else if( useCount < 0 ) 
		{
			throw std::runtime_error("vsDetachThread() called when already detached");
		}
	}
}

void VsEnvironManager::DisposeEnv()
{
	VsEnviron& env = GetStaticEnv();
	DWORD envThread = GetEnvThread();

	if (env && envThread == ::GetCurrentThreadId() )
	{
		::vsDispose(env);
		env = 0;
	}
}