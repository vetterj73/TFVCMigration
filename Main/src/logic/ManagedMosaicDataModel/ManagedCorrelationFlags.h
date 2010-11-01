#pragma once

#include "CorrelationFlags.h"

public ref class ManagedCorrelationFlags
{
public:
	ManagedCorrelationFlags(MosaicDM::CorrelationFlags *pCorrelationFlags)
	{
		_pCorrelationFlags = pCorrelationFlags;
	}

	bool GetTriggerToTrigger()
	{
		return _pCorrelationFlags->GetTriggerToTrigger();
	}
	
	void SetTriggerToTrigger(bool triggerToTrigger)
	{
		_pCorrelationFlags->SetTriggerToTrigger(triggerToTrigger);
	}

	bool GetCameraToCamera()
	{
		return _pCorrelationFlags->GetCameraToCamera();
	}
	
	void SetCameraToCamera(bool cameraToCamera)
	{
		_pCorrelationFlags->SetCameraToCamera(cameraToCamera);
	}

private:
	MosaicDM::CorrelationFlags *_pCorrelationFlags;
};

