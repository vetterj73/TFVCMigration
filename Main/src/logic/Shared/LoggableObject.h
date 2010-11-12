#pragma once
#include "windows.h"

namespace LOGGER
{
	enum LOGTYPE 
	{
		LogTypeError,
		LogTypeWarning,
		LogTypeDiagnostic,
		LogTypeSystem,
		NUMLOGTYPES
	};

	typedef void (*LOGGINGCALLBACK)(const char* message, LOGTYPE LogType);

	class LoggableObject
	{
	public:

		LoggableObject()
		{
			_registeredLoggingCallback = NULL;

			for(int i=0; i<(int)NUMLOGTYPES; i++)
				_logTypeFlags[i] = false;
		}

		// Register a function that will be used for log messages.
		void RegisterLoggingCallback(LOGGINGCALLBACK callback)
		{
			_registeredLoggingCallback = callback;
		}

		bool IsLoggingType(LOGTYPE logType)
		{
			return _logTypeFlags[logType];
		};

		void SetLogType(LOGTYPE logType, bool bOn)
		{
			_logTypeFlags[logType] = bOn;
		}
	
		// Let listener know that a log message they are interested in has occurred
		void FireLogEntry(const char *message, LOGTYPE logType)
		{
			if(_registeredLoggingCallback == NULL)
				return;

			if(IsLoggingType(logType))
				_registeredLoggingCallback(message, logType);
		}

	private:
		LOGGINGCALLBACK _registeredLoggingCallback;
		bool _logTypeFlags[(int)NUMLOGTYPES];
	};
}