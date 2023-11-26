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
			double dMinHeight=0);

		MaskInfo(
			bool bMask, 
			unsigned char* pBuf,
			int iColumns,
			int iRows,
			int iSpan,
			double dPixelSize);

		~MaskInfo();

		bool _bMask;
		double _dMinHeight;
		bool _bOnlyCalOveralpWithMask;	// Not support yet
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
		
		MaskInfo* GetMaskInfo(){return &_maskInfo;};
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