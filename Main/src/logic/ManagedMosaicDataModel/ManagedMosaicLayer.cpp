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
		ManagedFOVPreferSelected^ pPreferSelectedM)
	{
		MosaicDM::FOVPreferSelected preferSelectedFov;
		preferSelectedFov.preferLR = (MosaicDM::FOVLRPOS)pPreferSelectedM->preferLR;
		preferSelectedFov.preferTB = (MosaicDM::FOVTBPOS)pPreferSelectedM->preferTB;

		bool bFlag = _pMosaicLayer->GetImagePatch(
			(unsigned char*)(void*) pBuf,
			iPixelSpan,
			iStartRowInCad,
			iStartColInCad,
			iRows,
			iCols,
			&preferSelectedFov);

		pPreferSelectedM->selectedLR = (FOVLRPOSM)preferSelectedFov.selectedLR;
		pPreferSelectedM->selectedTB = (FOVTBPOSM)preferSelectedFov.selectedTB;

		return(bFlag);
	}
}
