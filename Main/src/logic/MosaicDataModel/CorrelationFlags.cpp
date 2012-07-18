#include "stdafx.h"
#include "CorrelationFlags.h"

namespace MosaicDM 
{
	MaskInfo::MaskInfo()
	{
		_bMask = false;
		_dMinHeight = -1;

		_pPanelMaskImage = NULL;
	}

	MaskInfo::MaskInfo(
		bool bMask, 
		double dMinHeight)
	{
		_bMask = bMask;
		_dMinHeight = dMinHeight;

		_pPanelMaskImage = NULL;
	}

	MaskInfo::MaskInfo(
		bool bMask, 
		unsigned char* pBuf,
		int iColumns,
		int iRows,
		int iSpan,
		double dPixelSize)
	{
		// create image transform
		double t[3][3];
		t[0][0] = dPixelSize;
		t[0][1] = 0;
		t[0][2] = 0;
		t[1][0] = 0;
		t[1][1] = dPixelSize;
		t[1][2] = 0;
		t[2][0] = 0;
		t[2][1] = 0;
		t[2][2] = 1;
		ImgTransform trans(t);

		unsigned int iBytePerPixel = 1;
		bool bCreateOwnBuf = false;
		_pPanelMaskImage = new Image(iColumns, iRows, iSpan, iBytePerPixel, 
			trans, trans, bCreateOwnBuf, pBuf);
	}

	MaskInfo::~MaskInfo()
	{
		if(_pPanelMaskImage != NULL)
			delete _pPanelMaskImage;
	}

	CorrelationFlags::CorrelationFlags(void)
	{
		_cameraToCamera = true;
		_triggerToTrigger = true;
		_applyCorrelationAreaSizeUpLimit = false;
	}
}