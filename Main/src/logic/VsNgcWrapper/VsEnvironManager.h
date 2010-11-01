#pragma once

#include <windows.h>
#include "vision.h"

class VsEnvironManager
{
public:

	static VsEnvironManager& Instance(void);

	static VsEnviron getEnv(); // Get environment for calling thread.
	static void releaseEnv();	// Match every call to get() with one to release().
	static void disposeEnv();

	// Pass environment point in before an inside environment is created
	bool SetEnv(VsEnviron& env, DWORD threadId);

protected:
	VsEnvironManager(void);
	~VsEnvironManager(void);

private:
	static VsEnviron& getStaticEnv(); // Get environment for calling thread.
	static DWORD&     getEnvThread(); // Get thread ID of the owner thread
	static int        getAddEnvUseCount(int add=0); // Get the use count for this thread and then add the supplied value
};

