#include "VsEnvironManager.h"

#include <map>
using std::map;

VsEnvironManager::VsEnvironManager(void)
{
}

VsEnvironManager::~VsEnvironManager(void)
{
}


VsEnviron& VsEnvironManager::getStaticEnv()
{
	static VsEnviron retVal = 0;
	return retVal;
}

DWORD& VsEnvironManager::getEnvThread()
{
	static DWORD retVal = 0;
	return retVal;
}

int VsEnvironManager::getAddEnvUseCount( int add/*=0*/ )
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


VsEnviron VsEnvironManager::getEnv()
{
	VsEnviron& retVal = getStaticEnv();

	if (!retVal)
	{
		retVal = ::vsCreateSharedVisionEnviron(0, 0);
		getEnvThread() = ::GetCurrentThreadId();
	}
	
	DWORD envThread = getEnvThread();

	if (envThread != ::GetCurrentThreadId() )
	{
		int useCount = getAddEnvUseCount(+1);
		if (useCount == 1)
		{
			if(vsAttachThread() == -1)
				throw std::runtime_error("vsAttachThread() failed");
		}
	}
	return retVal;
}


void VsEnvironManager::releaseEnv()
{
	DWORD envThread = getEnvThread();

	if (envThread != ::GetCurrentThreadId() )
	{
		int useCount = getAddEnvUseCount(-1);
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

void VsEnvironManager::disposeEnv()
{
	VsEnviron& env = getStaticEnv();
	DWORD envThread = getEnvThread();

	if (env && envThread == ::GetCurrentThreadId() )
	{
		::vsDispose(env);
		env = 0;
	}
}