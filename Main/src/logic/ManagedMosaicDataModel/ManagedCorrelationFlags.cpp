#include "StdAfx.h"
#include "ManagedCorrelationFlags.h"

namespace MMosaicDM 
{
	ManagedMaskInfo::ManagedMaskInfo()
	{
		_bMask = false;
		_dMinHeight = 0;
		_bMaskFirstLayer = true;
		_bOnlyCalOveralpWithMask = false;
	}

	ManagedMaskInfo::ManagedMaskInfo(
		bool bMask, 
		double dMinHeight,
		bool bMaskFirstLayer,
		bool bOnlyCalOveralpWithMask)
	{
		_bMask = bMask;
		_dMinHeight = dMinHeight;
		_bMaskFirstLayer = bMaskFirstLayer;
		_bOnlyCalOveralpWithMask = bOnlyCalOveralpWithMask;
	}

	void ManagedCorrelationFlags::SetMaskInfo(ManagedMaskInfo^ infoM)
	{
		MosaicDM::MaskInfo maskInfo;
		maskInfo._bMask = infoM->_bMask;
		maskInfo._dMinHeight = infoM->_dMinHeight;
		maskInfo._bMaskFirstLayer = infoM->_bMaskFirstLayer;
		maskInfo._bOnlyCalOveralpWithMask = infoM->_bOnlyCalOveralpWithMask;

		_pCorrelationFlags->SetMaskInfo(maskInfo);
	}

	ManagedMaskInfo^ ManagedCorrelationFlags::GetMaskInfo()
	{
		MosaicDM::MaskInfo* pMaskInfo = _pCorrelationFlags->GetMaskInfo();

		ManagedMaskInfo^ infoM = gcnew ManagedMaskInfo() ;
		infoM->_bMask = pMaskInfo->_bMask;
		infoM->_dMinHeight = pMaskInfo->_dMinHeight;
		infoM->_bMaskFirstLayer = pMaskInfo->_bMaskFirstLayer;
		infoM->_bOnlyCalOveralpWithMask = pMaskInfo->_bOnlyCalOveralpWithMask;

		return(infoM);
	}
}