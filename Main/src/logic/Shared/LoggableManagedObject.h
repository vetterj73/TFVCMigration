#pragma once
#include "LoggableObject.h"
using namespace System;
using namespace System::Runtime::InteropServices;
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

	protected delegate void LoggingDelegateForUnmanaged(const char *message, LOGTYPE logType);
	public delegate void LoggingDelegate(String ^ message, MLOGTYPE logType);

	public ref class MLoggableObject
	{
		public:
			MLoggableObject()
			{
				_pLoggableObject = NULL;
				_loggingDelegate = gcnew LoggingDelegateForUnmanaged(this, &MLOGGER::MLoggableObject::RaiseLogEntry); 
			}

			void SetLogType(MLOGTYPE logType, bool bOn)
			{
				if(_pLoggableObject)
					_pLoggableObject->SetLogType((LOGTYPE)logType, bOn);
			}

			event LoggingDelegate^ OnLogEntry;

		protected:
			void SetLoggableObject(LoggableObject *pLoggableObject)
			{
				_pLoggableObject = pLoggableObject;
				_pLoggableObject->RegisterLoggingCallback((LOGGINGCALLBACK)Marshal::GetFunctionPointerForDelegate(_loggingDelegate).ToPointer());
			}

			LoggingDelegateForUnmanaged ^_loggingDelegate;
			void RaiseLogEntry(const char* error, LOGTYPE logType)
			{
				String ^msg = gcnew String(error);
				OnLogEntry(msg, (MLOGTYPE)logType);
			}

			LoggableObject* _pLoggableObject;
	};
}