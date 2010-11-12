#pragma once
#include "stdlib.h" // Defines NULL

///
///	Simple Interface that allows log information to be "bubbled up" to some higher level
/// The upper level component can decide how to persist the log to disk
///
namespace LOGGER
{
	///
	/// Different types of things you may want to log...
	///
	enum LOGTYPE 
	{
		LogTypeError,
		LogTypeWarning,
		LogTypeDiagnostic,
		LogTypeSystem,
		NUMLOGTYPES
	};

	/// The one and only logging callback function.
	typedef void (*LOGGINGCALLBACK)(const char* message, LOGTYPE LogType);

	///
	///	This is a simple class that adds logging capabilities to any subclass.
	/// The main idea:  If you use this for logging, you allow some an upper level
	/// component to decide how the log is persisted.  This simply uses a callback 
	/// to log to "somewhere".
	///
	class LoggableObject
	{
	public:

		///
		///	Constructor - by default, all logging is disabled.  
		/// @todo - perhaps we want to enabled errors...
		///
		LoggableObject()
		{
			_registeredLoggingCallback = NULL;
			SetAllLogTypes(false);
		}

		///
		/// Register a function that will be used for log messages.
		///
		void RegisterLoggingCallback(LOGGINGCALLBACK callback)
		{
			_registeredLoggingCallback = callback;
		}

		///
		///	Checks if this log type is turned on (this allows the sub class to 
		/// only create log entries when necessary).  It makes sense to check
		/// if there is lots of info being logged.
		///
		bool IsLoggingType(LOGTYPE logType)
		{
			return _logTypeFlags[logType];
		};

		///
		///	Turns on/off a particular log type.
		///
		void SetLogType(LOGTYPE logType, bool bOn)
		{
			_logTypeFlags[logType] = bOn;
		}

		///
		///	Turns on or off all log types.
		///
		void SetAllLogTypes(bool bOn)
		{
			for(int i=0; i<(int)NUMLOGTYPES; i++)
				SetLogType((LOGTYPE)i, bOn);
		}

		///
		/// Lets the listener know that an item is ready to be logged.
		///
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