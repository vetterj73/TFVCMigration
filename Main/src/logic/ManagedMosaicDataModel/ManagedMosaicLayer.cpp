#include "StdAfx.h"
#include "ManagedMosaicLayer.h"


namespace MMosaicDM 
{
	bool ManagedMosaicLayer::GetImagePatch(
		System::IntPtr pBuf,
		unsigned int iPixelSpan,
		unsigned int iStartCol,
		unsigned int iWidth,
		unsigned int iStartRow,
		unsigned int iHeight,
		ManagedFOVPreferSelected preferSelectedM)
	{
		MosaicDM::FOVPreferSelected preferSelectedFov;
		preferSelectedFov.preferLR = preferSelectedM.preferLR;
		preferSelectedFov.preferTB = preferSelectedM.preferTB;

		bool bFlag = _pMosaicLayer->GetImagePatch(
			(unsigned char*)(void*) pBuf,
			iPixelSpan,
			iStartCol,
			iWidth,
			iStartRow,
			iHeight,
			&preferSelectedFov);

		preferSelectedM.selectedLR = preferSelectedFov.selectedLR;
		preferSelectedM.selectedTB = preferSelectedFov.selectedTB;

		return(bFlag);
	}

}


