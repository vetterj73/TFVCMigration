#include "stdafx.h"
#include "CorrelationFlags.h"

namespace MosaicDM 
{
	MaskInfo::MaskInfo()
	{
		_bMask = false;
		_dMinHeight = 0;
		_bMaskFirstLayer = true;
		_bOnlyCalOveralpWithMask = false;
	}

	MaskInfo::MaskInfo(
		bool bMask, 
		double dMinHeight,
		bool bMaskFirstLayer,
		bool bOnlyCalOveralpWithMask)
	{
		_bMask = bMask;
		_dMinHeight = dMinHeight;
		_bMaskFirstLayer = bMaskFirstLayer;
		_bOnlyCalOveralpWithMask = bOnlyCalOveralpWithMask;
	}

	CorrelationFlags::CorrelationFlags(void)
	{
		_cameraToCamera = true;
		_triggerToTrigger = true;
		_applyCorrelationAreaSizeUpLimit = false;
	}
}