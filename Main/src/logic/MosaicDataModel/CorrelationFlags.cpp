#include "stdafx.h"
#include "CorrelationFlags.h"

namespace MosaicDM 
{
	MaskInfo::MaskInfo()
	{
		_bMask = false;
		_dMinHeight = -1;
		_bMaskFirstLayer = true;
		_bOnlyCalOveralpWithMask = false;

		_iPanelMaskIndex = -1;
		_pPanelMaskImage = NULL;
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

		_iPanelMaskIndex = -1;
		_pPanelMaskImage = NULL;
	}

	CorrelationFlags::CorrelationFlags(void)
	{
		_cameraToCamera = true;
		_triggerToTrigger = true;
		_applyCorrelationAreaSizeUpLimit = false;
	}
}