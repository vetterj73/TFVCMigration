#pragma once
#include "Image.h"

namespace MosaicDM 
{
	class  MaskInfo
	{
	public:
		MaskInfo();
		MaskInfo(
			bool bMask, 
			double dMinHeight,
			bool bMaskFirstLayer,
			bool bOnlyCalOveralpWithMask);

		bool _bMask;
		double _dMinHeight;
		bool _bMaskFirstLayer;
		bool _bOnlyCalOveralpWithMask;
		int _iPanelMaskIndex;
		Image* _pPanelMaskImage;
	};

	///
	///	Correlation Flags are the types of correlations that will be performed for each tile in each layer
	///
	class CorrelationFlags
	{
	public:
		CorrelationFlags(void);

		bool GetCameraToCamera(){return _cameraToCamera;};
		void SetCameraToCamera(bool cToc){_cameraToCamera = cToc;};

		bool GetTriggerToTrigger(){return _triggerToTrigger;};
		void SetTriggerToTrigger(bool tTot){_triggerToTrigger = tTot;};		
		
		MaskInfo GetMaskInfo(){return _maskInfo;};
		void SetMaskInfo(MaskInfo maskInfo){_maskInfo = maskInfo;};

		bool GetApplyCorrelationAreaSizeUpLimit(){return _applyCorrelationAreaSizeUpLimit;};
		void SetApplyCorrelationAreaSizeUpLimit(bool ApplyCorrelationAreaSizeUpLimit){_applyCorrelationAreaSizeUpLimit = ApplyCorrelationAreaSizeUpLimit;};

	private:
		bool _cameraToCamera;
		bool _triggerToTrigger;

		MaskInfo _maskInfo;
		
		bool _applyCorrelationAreaSizeUpLimit;
	};
}