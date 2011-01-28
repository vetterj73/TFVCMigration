// This is the main DLL file.

#include "stdafx.h"

#include "ManagedPanelAlignment.h"
using namespace System;
using namespace System::Runtime::InteropServices;

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
		_pAligner = NULL;
	}

	ManagedPanelAlignment::~ManagedPanelAlignment()
	{
		if(_pAligner!=NULL)
			delete _pAligner;

		_pAligner = NULL;
	}

	// Change production
	bool ManagedPanelAlignment::ChangeProduction(ManagedMosaicSet^ set, CPanel^ panel, unsigned int numThreads)
	{
		MosaicSet* pMosaicSet = (MosaicSet*)(void*)set->UnmanagedMosaicSet;
		Panel* pPanel  = (Panel*)(void*)panel->UnmanagedPanel;

		bool bFlag = _pAligner->ChangeProduction(pMosaicSet, pPanel, numThreads);

		return(bFlag);
	}

	// Reset for next panel
	void ManagedPanelAlignment::ResetForNextPanel()
	{
		_pAligner->ResetForNextPanel();
	}

	bool ManagedPanelAlignment::Save3ChannelImage(System::String^ imagePath,
		System::IntPtr pChannel1, 
		System::IntPtr pChannel2,	
		System::IntPtr pChannel3, 
		int numRows, int numColumns)
	{
		System::IntPtr stringPtr = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(imagePath);
		std::string nativeImagePath = (char*)stringPtr.ToPointer();			

		bool bSaved = _pAligner->Save3ChannelImage(nativeImagePath, 
			(unsigned char*)(void*)pChannel1,
			(unsigned char*)(void*)pChannel2,
			(unsigned char*)(void*)pChannel3,
			numRows, numColumns);

		System::Runtime::InteropServices::Marshal::FreeHGlobal(stringPtr);
		return bSaved;
	}
}