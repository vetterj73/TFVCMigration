#include "StdAfx.h"
#include "ManagedMosaicLayer.h"


namespace MMosaicDM 
{
	bool ManagedMosaicLayer::GetImagePatch(
		System::IntPtr pBuf,
		unsigned int iPixelSpan,
		unsigned int iStartRowInCad,
		unsigned int iStartColInCad,
		unsigned int iRows,
		unsigned int iCols,
		ManagedFOVPreferSelected preferSelectedM)
	{
		MosaicDM::FOVPreferSelected preferSelectedFov;
		preferSelectedFov.preferLR = preferSelectedM.preferLR;
		preferSelectedFov.preferTB = preferSelectedM.preferTB;

		bool bFlag = _pMosaicLayer->GetImagePatch(
			(unsigned char*)(void*) pBuf,
			iPixelSpan,
			iStartRowInCad,
			iStartColInCad,
			iRows,
			iCols,
			&preferSelectedFov);

		preferSelectedM.selectedLR = preferSelectedFov.selectedLR;
		preferSelectedM.selectedTB = preferSelectedFov.selectedTB;

		return(bFlag);
	}
}
