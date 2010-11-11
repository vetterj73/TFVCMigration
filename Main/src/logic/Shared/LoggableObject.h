#pragma once

typedef void (*LOGGINGCALLBACK)(const char* message);

#include "windows.h"

class LoggableObject
{
public:
	LoggableObject()
	{
		registeredDiagnosticMessageCallback_ = NULL;
		registeredErrorCallback_ = NULL;
	}

	// Register a function that will be used for diagnostic messages.
	void RegisterDiagnosticMessageCallback(LOGGINGCALLBACK callback)
	{
		registeredDiagnosticMessageCallback_ = callback;
	}
	
	// Register a function that will be used for error messages.
	void RegisterErrorCallback(LOGGINGCALLBACK callback)
	{
		registeredErrorCallback_ = callback;
	}

	// Let listener know that a Diagnostic Message Occurred
	void FireDiagnosticMessage(const char *message)
	{
		if(registeredDiagnosticMessageCallback_ != NULL)
			registeredDiagnosticMessageCallback_(message);
	}

	// Let listener know that an Error occurred
	void FireError(const char *message)
	{
		if(registeredErrorCallback_ != NULL)
			registeredErrorCallback_(message);
	}

	// Let listener know if logging is enabled.
	bool LogDiagnostics(){return registeredDiagnosticMessageCallback_!=NULL;};
	bool LogErrors(){return registeredErrorCallback_!=NULL;};

private:
	LOGGINGCALLBACK registeredDiagnosticMessageCallback_;
	LOGGINGCALLBACK registeredErrorCallback_;
};
