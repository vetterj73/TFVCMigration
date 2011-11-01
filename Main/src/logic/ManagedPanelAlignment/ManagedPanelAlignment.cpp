// This is the main DLL file.

#include "stdafx.h"

#include "ManagedPanelAlignment.h"
#include "FiducialResult.h"
#include "EquationWeights.h"
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

	// Report fiducial results for a panel back
	// Should be called afer a panel is stitched and before application is reset for next panel 
	ManagedPanelFidResultsSet^ ManagedPanelAlignment::GetFiducialResultsSet()
	{
		PanelFiducialResultsSet* resultsSet= _pAligner->GetFidResultsSetPoint();
		ManagedPanelFidResultsSet ^mSet = gcnew ManagedPanelFidResultsSet;
		mSet->resultsSet = gcnew List<ManagedPanelFidResults^>;

		for(int i=0; i<resultsSet->Size(); i++)
		{
			// for each physical fiducial
			PanelFiducialResults* pResults = resultsSet->GetPanelFiducialResultsPtr(i);
			ManagedPanelFidResults ^mResults = gcnew ManagedPanelFidResults;
			mResults->results = gcnew List<ManagedFidResult^>;

			mResults->iID = pResults->GetId();
			mResults->dCadX = pResults->GetCadX();
			mResults->dCadY = pResults->GetCadY();
			mResults->dConfidence = pResults->CalConfidence();

			// for each fiducial overlap
			list<FidFovOverlap*>* pResultsList = pResults->GetFidOverlapListPtr();
			for(list<FidFovOverlap*>::iterator j = pResultsList->begin(); j != pResultsList->end(); j++)
			{
				// FOV information
				ManagedFidResult ^mResult = gcnew ManagedFidResult();
				mResult->iLayerIndex = (*j)->GetMosaicImage()->Index();
				mResult->iTrigIndex = (*j)->GetTriggerIndex();
				mResult->iCamIndex = (*j)->GetCameraIndex();

				// Correlation results
				mResult->rowOffset = (*j)->GetCoarsePair()->GetCorrelationResult().RowOffset;
				mResult->colOffset = (*j)->GetCoarsePair()->GetCorrelationResult().ColOffset;
				mResult->correlationScore = (*j)->GetCoarsePair()->GetCorrelationResult().CorrCoeff;
				mResult->ambiguityScore = (*j)->GetCoarsePair()->GetCorrelationResult().AmbigScore;
				
				// weight for solver
				mResult->weight = Weights.CalWeight((*j)->GetCoarsePair());
				
				mResults->results->Add(mResult);
			}
			
			mSet->resultsSet->Add(mResults);
			mSet->dConfidence = resultsSet->CalConfidence();
		}

		return mSet;
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

	void ManagedPanelAlignment::LogTransformVectors(bool bLog)	
	{
		_pAligner->LogTransformVectors(bLog);
	}

	void ManagedPanelAlignment::FiducialSearchExpansionXInMeters(double fidSearchXInMeters)
	{
		_pAligner->FiducialSearchExpansionXInMeters(fidSearchXInMeters);
	}
	
	void ManagedPanelAlignment::FiducialSearchExpansionYInMeters(double fidSearchYInMeters)
	{
		_pAligner->FiducialSearchExpansionYInMeters(fidSearchYInMeters);
	}

	void ManagedPanelAlignment::UseCyberNgc4Fiducial()
	{
		_pAligner->UseCyberNgc4Fiducial();
	}

	void ManagedPanelAlignment::UseProjectiveTransform(bool bValue)
	{
		_pAligner->UseProjectiveTransform(bValue);
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

	bool ManagedPanelAlignment::Save3ChannelImage(System::String^ imagePath,
		System::IntPtr pChannel1, int iSpan1, 
		System::IntPtr pChannel2, int iSpan2,
		System::IntPtr pChannel3, int iSpan3,
		int numRows, int numColumns)
	{
		System::IntPtr stringPtr = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(imagePath);
		std::string nativeImagePath = (char*)stringPtr.ToPointer();			

		bool bSaved = _pAligner->Save3ChannelImage(nativeImagePath, 
			(unsigned char*)(void*)pChannel1, iSpan1,
			(unsigned char*)(void*)pChannel2, iSpan2,
			(unsigned char*)(void*)pChannel3, iSpan3,
			numRows, numColumns);

		System::Runtime::InteropServices::Marshal::FreeHGlobal(stringPtr);
		return bSaved;
	}
}