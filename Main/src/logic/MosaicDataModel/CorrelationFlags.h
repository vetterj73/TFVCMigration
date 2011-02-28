#pragma once

namespace MosaicDM 
{
	///
	///	Correlation Flags are the types of correlations that will be performed for each tile in each layer
	///
	class CorrelationFlags
	{
	public:
		CorrelationFlags(void)
		{
			_cameraToCamera = true;
			_triggerToTrigger = true;
			_maskNeeded = false;
			_applyCorrelationAreaSizeUpLimit = false;
		}

		bool GetCameraToCamera(){return _cameraToCamera;};
		void SetCameraToCamera(bool cToc){_cameraToCamera = cToc;};

		bool GetTriggerToTrigger(){return _triggerToTrigger;};
		void SetTriggerToTrigger(bool tTot){_triggerToTrigger = tTot;};		
		
		bool GetMaskNeeded(){return _maskNeeded;};
		void SetMaskNeeded(bool maskNeeded){_maskNeeded = maskNeeded;};

		bool GetApplyCorrelationAreaSizeUpLimit(){return _applyCorrelationAreaSizeUpLimit;};
		void SetApplyCorrelationAreaSizeUpLimit(bool ApplyCorrelationAreaSizeUpLimit){_applyCorrelationAreaSizeUpLimit = ApplyCorrelationAreaSizeUpLimit;};

	private:
		bool _cameraToCamera;
		bool _triggerToTrigger;
		bool _maskNeeded;
		bool _applyCorrelationAreaSizeUpLimit;
	};
}