#pragma once
#include "LoggableObject.h"
using namespace System;
using namespace System::Runtime::InteropServices;

///
///	This is used as a helper class to bubble up logging information to the managed layer.
///
namespace MLOGGER
{
	///
	///	NOTE:  These must stay consistant with what is in the Unmanaged LoggableObject.
	///
	public enum class MLOGTYPE 
	{
		LogTypeError,
		LogTypeWarning,
		LogTypeDiagnostic,
		LogTypeSystem,
		NUMLOGTYPES
	};

	///
	///	Delegate - unmanaged to managed.
	///
	protected delegate void LoggingDelegateForUnmanaged(const char *message, LOGTYPE logType);
	
	///
	///	Delegate to managed world.
	///
	public delegate void LoggingDelegate(String ^ message, MLOGTYPE logType);

	///
	///	Wrapper for an unmanaged LoggableObject
	///
	public ref class MLoggableObject
	{
		public:
			///
			///	Constructor - Hooks up the unmanaged delegate
			///
			MLoggableObject()
			{
				_pLoggableObject = NULL;
				_loggingDelegate = gcnew LoggingDelegateForUnmanaged(this, &MLOGGER::MLoggableObject::RaiseLogEntry); 
			}

			///
			/// Turns on/off a particular log type.
			///
			void SetLogType(MLOGTYPE logType, bool bOn)
			{
				if(_pLoggableObject)
					_pLoggableObject->SetLogType((LOGTYPE)logType, bOn);
			}

			///
			///	Turns on/off all log types.
			///
			void SetAllLogTypes(bool bOn)
			{
				if(_pLoggableObject)
					_pLoggableObject->SetAllLogTypes(bOn);	
			}

			///
			///	Managed Event Delegate
			///
			event LoggingDelegate^ OnLogEntry;

		protected:

			///
			/// Unmanaged Loggable Object...
			///
			LoggableObject* _pLoggableObject;
			
			///
			///	This allows the managed class to hook in to the unmanaged loggable object.
			///
			void SetLoggableObject(LoggableObject *pLoggableObject)
			{
				_pLoggableObject = pLoggableObject;
				_pLoggableObject->RegisterLoggingCallback((LOGGINGCALLBACK)Marshal::GetFunctionPointerForDelegate(_loggingDelegate).ToPointer());
			}

			///
			/// Plumbing from unmanaged to managed.
			///
			LoggingDelegateForUnmanaged ^_loggingDelegate;
			void RaiseLogEntry(const char* error, LOGTYPE logType)
			{
				String ^msg = gcnew String(error);
				OnLogEntry(msg, (MLOGTYPE)logType);
			}
	};
}