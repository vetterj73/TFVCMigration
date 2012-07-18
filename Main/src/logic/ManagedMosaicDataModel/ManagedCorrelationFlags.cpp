#include "StdAfx.h"
#include "ManagedCorrelationFlags.h"

namespace MMosaicDM 
{
	ManagedMaskInfo::ManagedMaskInfo()
	{
		_bMask = false;
		_dMinHeight = 0;
	}

	ManagedMaskInfo::ManagedMaskInfo(
		bool bMask, 
		double dMinHeight)
	{
		_bMask = bMask;
		_dMinHeight = dMinHeight;;
	}

	void ManagedCorrelationFlags::SetMaskInfo(ManagedMaskInfo^ infoM)
	{
		MosaicDM::MaskInfo maskInfo;
		maskInfo._bMask = infoM->_bMask;
		maskInfo._dMinHeight = infoM->_dMinHeight;

		_pCorrelationFlags->SetMaskInfo(maskInfo);
	}

	ManagedMaskInfo^ ManagedCorrelationFlags::GetMaskInfo()
	{
		MosaicDM::MaskInfo* pMaskInfo = _pCorrelationFlags->GetMaskInfo();

		ManagedMaskInfo^ infoM = gcnew ManagedMaskInfo() ;
		infoM->_bMask = pMaskInfo->_bMask;
		infoM->_dMinHeight = pMaskInfo->_dMinHeight;

		return(infoM);
	}
}