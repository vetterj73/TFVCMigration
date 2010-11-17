// This is the main DLL file.

#include "stdafx.h"

#include "ManagedPanelAlignment.h"

namespace PanelAlignM {

	ManagedPanelAlignment::ManagedPanelAlignment()
	{
		_pAligner = NULL;
	}

	ManagedPanelAlignment::!ManagedPanelAlignment()
	{
		if(_pAligner!=NULL)
			delete _pAligner;
	}

	bool ManagedPanelAlignment::SetPanel(ManagedMosaicSet^ set, CPanel^ panel)
	{
		if(_pAligner!=NULL)
		{
			delete _pAligner;
			_pAligner = NULL;
		}

		_pAligner = new PanelAligner();

		MosaicSet* pMosaicSet = (MosaicSet*)(void*)set->UnmanagedMosaicSet;
		Panel* pPanel  = (Panel*)(void*)panel->UnmanagedPanel;

		bool bFlag = _pAligner->SetPanel(pMosaicSet, pPanel);

		return(bFlag);
	}

	bool ManagedPanelAlignment::AddImage(
			int iLayerIndex, 
			int iTrigIndex, 
			int iCamIndex)
	{
		bool bFlag = _pAligner->AddImage(
			(unsigned int)iLayerIndex, 
			(unsigned int)iTrigIndex, 
			(unsigned int)iCamIndex);

		return(bFlag);
	}

}