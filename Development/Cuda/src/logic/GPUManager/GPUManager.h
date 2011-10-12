#pragma once

#include "windows.h"
#include <queue>
#include <string>
using std::queue;
using std::string;
using std::vector;

// unsigned char matrix structure
typedef struct {
    unsigned int width;
    unsigned int height;
    unsigned int size;
    unsigned char* elements;
} ByteMatrix;

typedef unsigned long GPUSTATUS;
typedef unsigned long SESSIONHANDLE;

	/// \mainpage
	///
	/// \section intro Introduction
	/// This reference document describes the SIM sensor native API. Simulation can be enabled via XML
	/// by using the SetSimulationFile function to set a valid xml path. Recording can be enabled by
	/// specifying a folder to record to.
	/// \n\n
	/// \image html SIM.png CyberOptics' SIM Sensor
	/// \image latex SIM.png "CyberOptics' SIM Sensor" width=10cm
	///
	/// \section org Software Interface
	/// CyberOptics' SIM Native CoreAPI is accessed from the \ref SIMAPI namespace.  
	/// \n\n
	/// The class \ref SIMAPI::SIMCore provides a static interface to initialize and access
	/// the CoreAPI methods, properties, and events. This class also provices a list of the
	/// ISIMDevice objects in the system.
	/// \n\n
	/// The interface class \ref SIMAPI::ISIMDevice provides access to a specific SIM Sensor
	/// attached to the computer. The interface also provices a list of ISIMCamera objects in
	/// the device.
	/// \n\n
	/// The interface class \ref SIMAPI::ISIMCamera provides access to a specific camera in
	/// the SIM sensor. It camera object is accessed from the parent ISIMDevice.
	/// \n\n
	/// The interface class \ref SIMAPI::CSIMCaptureSpec is used to describe the panel image
	/// acquisition procedure for a specific ISIMDevice. This is a description of (1) the camera
	/// bar trigger positions with respect to the panel home position. The trigger position describe
	/// where the the camera bar will acquire an image rows as the panel travels beneath the SIM sensor.
	/// The capture spec also describes the illumination settings used to acquire those images.
	/// \n\n
	/// The interface class \ref SIMAPI::CSIMIllumination describes the illumination properties
	/// for image acquisition.
	/// \n\n
	/// The interface class \ref SIMAPI::CSIMFrame describes a field of view image. This includes
	/// the position of the FOV within a panel image and the FOV image buffer pointer. A frame object is
	/// passed to a delegate instantiated by the application which recieves camera images during a panel
	/// acquisition cycle.
	/// \n\n\n\n
	///
	/// \latexonly	\pagebreak \endlatexonly
	///
	/// \section blockDiagram Software Block Diagram
	/// The block diagram below illustrates the relationships between the major CoreAPI classes.
	/// \image html ManagedCoreAPI.png CyberOptics' CoreAPI block diagram
	/// \image latex ManagedCoreAPI.png "CyberOptics' CoreAPI block diagram" width=15cm
	///
	/// \n\n
	///
	/// \latexonly	\pagebreak \endlatexonly
	///
	/// \section coordinateSystem Coordinate System
	/// The figure below illustrates the SIM sensor coordinate system used by all ManagedSIMDevice
	/// properties. For all properties, units of distance are in meters and units of rotation are in radians.
	/// \image html System.png "CyberOptics' CoreAPI coordinate system" width=15cm
	/// \image latex System.png "CyberOptics' CoreAPI coordinate system" width=15cm
	///
	/// \n\n
	/// \section revisions Revision History
	/// \subsection Rev00 Version 00 (April 11, 2010 - Mike H.)
	/// Initial doxygen draft.
	///
	/// This documentation has been generated using Doxygen.
	///
	/// \page errorCodes Error Codes
	/// \section SIMStatus SIMStatus Error Codes
	/// \include SIMStatus.h
	///
	/// \namespace SIMAPI
	///
	/// \brief
	/// CyberOptics SIM native CoreAPI classes are accessed from the SIMAPI namespace.
	///

namespace CyberGPU
{
	class GPUManager;
	//class IGPUJob;
	class GPUJobThread;
	class GPUStream;

	///
	///	Interface for a job (a task to perform on a separate thread).
	///
	class CGPUJob
	{
		friend GPUManager;

	public:
		enum GPUJobStatus
		{
			IDLE,
			ACTIVE,
			WAITING,
			COMPLETED,
		};

		CGPUJob();
		~CGPUJob();

		virtual void Run()=0;
		virtual GPUJobStatus GPURun(GPUStream *jobStream)=0; // returns job status after function execution

		HANDLE DoneEvent() { return _hEvent; }

		//virtual unsigned int NumberOfStreams()=0;

	protected:
	//public:

		GPUSTATUS SetDoneEvent();
		GPUSTATUS ResetDoneEvent();

	private:

		HANDLE _hEvent;

	};


	//typedef void (*ACQ_CALLBACK)(int device, SIMSTATUS status, CSIMFrame* frame, void* context);
	//typedef void (*ACQDONE_CALLBACK)(int device, SIMSTATUS status, int frameCount, void* context);
	//typedef void (*ERR_CALLBACK)(int device, SIMSTATUS status, void* context);
	//typedef void (*ARMED_CALLBACK)(int device, SIMSTATUS status, void* context);
	//typedef void (*INIT_CALLBACK)(SIMSTATUS status, int progress, void* context);
	//typedef void (*PROGRESS_CALLBACK)(int device, SIMSTATUS status, int progress, void* context);

	//class CSIMSimulationFile;
	//class ISIMCamera;
	//class ISIMDevice;
	//class ISIMDeviceGroup;

	/// \brief
	/// The SIMCore class is the top level class. It provides static access to the API.
	///
	/// The SIMCore class contains methods used to initialize and terminate the API. It also
	/// provides access to the system properties and a list of the ISIMDevice objects for the
	/// SIM devices attached to the computer.
	class GPUManager
	{

	private:
		// Don't allow construction from the outside - this is a singleton!
		GPUManager();
	public:
		~GPUManager();

		/// @cond MANUFACTURING

		static GPUManager&			Singleton();

		static GPUSTATUS			CheckGPUManager();
		//static SIMSTATUS			InitializeGPUManager();
		//static SIMSTATUS			RemoveGPUManager( void );

		static SESSIONHANDLE		CreateSession(CGPUJob *job);
		static void					CloseSession();

		static GPUSTATUS			RunJob(CGPUJob* pJob, SESSIONHANDLE hSession=NULL);
		static GPUSTATUS			RunJobAsynch(CGPUJob* pJob, SESSIONHANDLE hSession=NULL);

		DWORD						RunGPUThread();

	private:
		GPUSTATUS					GPUManagerStatus();
		GPUSTATUS					ConstructGPUManager();

		GPUSTATUS					QueueJob(CGPUJob* pJob, SESSIONHANDLE hSession);
		CGPUJob*					GetNextJob();
		void						ManageStreams();

		static GPUManager*			_instance;

		unsigned int				_maxStreams;

#pragma warning(push)
#pragma warning(disable:4251)
		HANDLE						_GPUThread;
		HANDLE						_startSignal;
		HANDLE						_killSignal;
		bool						_killThread;

		//vector<GPUJobThread*>		_jobThreads;
		vector<GPUStream*>		_jobStreams;

		HANDLE						_queueMutex;
		queue<CGPUJob*>				_jobQueue;

		vector<SESSIONHANDLE>		_SessionHandles;
		vector<void*>				_SessionContexts;

#pragma warning(pop)
	};

	bool CudaBufferRegister(unsigned char *ptr, size_t size);
	bool CudaBufferUnregister(unsigned char *ptr);
}
