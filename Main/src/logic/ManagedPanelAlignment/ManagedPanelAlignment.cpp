// This is the main DLL file.

#include "stdafx.h"

#include "ManagedPanelAlignment.h"

namespace PanelAlignM {

	ManagedPanelAlignment::ManagedPanelAlignment()
	{
		_pAligner = new PanelAligner();

		SetLoggableObject((System::IntPtr)(void*)_pAligner->GetLogger());
	}

	ManagedPanelAlignment::!ManagedPanelAlignment()
	{
		if(_pAligner!=NULL)
			delete _pAligner;
	}

	// Change production
	bool ManagedPanelAlignment::ChangeProduction(ManagedMosaicSet^ set, CPanel^ panel)
	{
		MosaicSet* pMosaicSet = (MosaicSet*)(void*)set->UnmanagedMosaicSet;
		Panel* pPanel  = (Panel*)(void*)panel->UnmanagedPanel;

		bool bFlag = _pAligner->ChangeProduction(pMosaicSet, pPanel);

		return(bFlag);
	}

	// Reset for next panel
	void ManagedPanelAlignment::ResetForNextPanel()
	{
		_pAligner->ResetForNextPanel();
	}

	// Add a image
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