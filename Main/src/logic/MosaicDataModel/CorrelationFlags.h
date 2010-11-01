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
		}

		bool GetCameraToCamera(){return _cameraToCamera;};
		void SetCameraToCamera(bool cToc){_cameraToCamera = cToc;};

		bool GetTriggerToTrigger(){return _triggerToTrigger;};
		void SetTriggerToTrigger(bool tTot){_triggerToTrigger = tTot;};

	private:
		bool _cameraToCamera;
		bool _triggerToTrigger;
	};
}