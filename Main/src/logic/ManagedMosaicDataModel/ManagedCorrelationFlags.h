#pragma once

#include "CorrelationFlags.h"

namespace MMosaicDM 
{

public ref class ManagedMaskInfo
{
public:
	ManagedMaskInfo();

	ManagedMaskInfo(
		bool bMask, 
		double dMinHeight,
		bool bMaskFirstLayer,
		bool bOnlyCalOveralpWithMask);

	bool _bMask;
	double _dMinHeight;
	bool _bMaskFirstLayer;
	bool _bOnlyCalOveralpWithMask;
};

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

	ManagedMaskInfo^ GetMaskInfo();
	
	void SetMaskInfo(ManagedMaskInfo^ infoM);

	// When ApplyCorrelationAreaSizeUpLimit == true; (default = false)
	// If the size of correlation pair is bigger than an internal defined size,
	// only the area with intern define size will be used for correlation calculation
	// This setting is used to speed up the stitching process.
	// However, sometime it will make the stitching for bare panel without paste unreliable
	bool GetApplyCorrelationAreaSizeUpLimit()
	{
		return _pCorrelationFlags->GetApplyCorrelationAreaSizeUpLimit();
	}
	
	void SetApplyCorrelationAreaSizeUpLimit(bool ApplyCorrelationAreaSizeUpLimit)
	{
		_pCorrelationFlags->SetApplyCorrelationAreaSizeUpLimit(ApplyCorrelationAreaSizeUpLimit);
	};

private:
	MosaicDM::CorrelationFlags *_pCorrelationFlags;
};

}

