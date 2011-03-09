// This is the main DLL file.

#include "stdafx.h"

#include "ManagedPanelAlignment.h"
using namespace System;
using namespace System::Runtime::InteropServices;

namespace PanelAlignM {

	ManagedPanelAlignment::ManagedPanelAlignment()
	{
		_pAligner = new PanelAligner();
		_pixelSizeX=1.70e-5;
		_pixelSizeY=1.70e-5;
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

	int ManagedPanelAlignment::GetNumberOfFidsProcessed()
	{
		return (int)_pAligner->GetLastProcessedFids()->size();
	}

	ManagedFidInfo^ ManagedPanelAlignment::GetFidAtIndex(unsigned int index)
	{
		if(index >= _pAligner->GetLastProcessedFids()->size())
			return nullptr;

		ManagedFidInfo ^fidM = nullptr;
		unsigned int count = 0;
		FidFovOverlapList* pFidFovList = _pAligner->GetLastProcessedFids();
		for(FidFovOverlapListIterator ite = pFidFovList->begin(); ite != pFidFovList->end(); ite++)
		{
			if(count == index)
			{
				FidFovOverlap fid = *ite;		
				fidM = gcnew ManagedFidInfo(fid.GetFiducialXPos(), fid.GetFiducialYPos(), 
					fid.GetCoarsePair()->GetCorrelationResult().RowOffset*_pixelSizeX, 
					fid.GetCoarsePair()->GetCorrelationResult().ColOffset*_pixelSizeY, 
					fid.GetCoarsePair()->GetCorrelationResult().CorrCoeff,
					fid.GetCoarsePair()->GetCorrelationResult().AmbigScore);
				break;
			}
			count++;
		}
		return fidM;
	}

	// Change production
	bool ManagedPanelAlignment::ChangeProduction(ManagedMosaicSet^ set, CPanel^ panel)
	{
		MosaicSet* pMosaicSet = (MosaicSet*)(void*)set->UnmanagedMosaicSet;
		Panel* pPanel  = (Panel*)(void*)panel->UnmanagedPanel;

		_pixelSizeX = set->GetNominalPixelSizeX();
		_pixelSizeY = set->GetNominalPixelSizeY();
		bool bFlag = _pAligner->ChangeProduction(pMosaicSet, pPanel);
		return(bFlag);
	}

	void ManagedPanelAlignment::NumThreads(unsigned int numThreads)
	{
		_pAligner->NumThreads(numThreads);
	}

	void ManagedPanelAlignment::LogFiducialOverlaps(bool bLog)
	{
		_pAligner->LogFiducialOverlaps(bLog);
	}
	
	void ManagedPanelAlignment::LogOverlaps(bool bLog)
	{
		_pAligner->LogOverlaps(bLog);
	}

	void ManagedPanelAlignment::LogMaskVectors(bool bLog)	
	{
		_pAligner->LogMaskVectors(bLog);
	}

	void ManagedPanelAlignment::FiducialSearchExpansionXInMeters(double fidSearchXInMeters)
	{
		_pAligner->FiducialSearchExpansionXInMeters(fidSearchXInMeters);
	}
	
	void ManagedPanelAlignment::FiducialSearchExpansionYInMeters(double fidSearchYInMeters)
	{
		_pAligner->FiducialSearchExpansionYInMeters(fidSearchYInMeters);
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