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
		_pixelSizeX=0; // avoid hard coded values, caller must initialize
		_pixelSizeY=0;
		SetLoggableObject((System::IntPtr)(void*)_pAligner->GetLogger());

		_alignmentDoneDelegate = gcnew AlignmentDoneDelegate(this, &ManagedPanelAlignment::RaiseAlignmentDone); 
		_pAligner->RegisterAlignmentDoneCallback((ALIGNMENTDONE_CALLBACK)Marshal::GetFunctionPointerForDelegate(_alignmentDoneDelegate).ToPointer(), NULL);	
		_pAligner->GetLogger()->SetAllLogTypes(true);
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

		// for each physical fiducial
		for(int i=0; i<resultsSet->Size(); i++)
		{
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
				mResult->iLayerIndex = (*j)->GetMosaicLayer()->Index();
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
		}

		mSet->dPanelSkew   = resultsSet->GetPanelSkew();
		mSet->dPanelXscale = resultsSet->GetXscale();
		mSet->dPanelYscale = resultsSet->GetYscale();
		
			
		// Confidence for overall panel
		mSet->dOverallConfidence = resultsSet->CalConfidence();
		// Confidence for each device
		unsigned int iNumDevice = _pAligner->GetMosaicSet()->GetNumDevice();
		mSet->dDeviceConfidences = gcnew array<double>((int)iNumDevice);
		for(unsigned int i=0; i<iNumDevice; i++)
			mSet->dDeviceConfidences[i] = resultsSet->CalConfidence(i);

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

	bool ManagedPanelAlignment::AddQXImageTile(System::IntPtr pbBuf, unsigned int iLayer, unsigned int iTrig, unsigned int iCam)	
	{
		_pAligner->AddQXImageTile((unsigned char*)(void*)pbBuf, iLayer, iTrig, iCam);
		return true;
	}
	
	bool ManagedPanelAlignment::GetQXTileTransform(unsigned int iLayer, unsigned int iTrig, unsigned int iCam, double dTrans[9])	
	{
		_pAligner->GetQXTileTransform(iLayer, iTrig, iCam, dTrans);
		return true;
	}

	bool ManagedPanelAlignment::ChangeQXproduction(
			double dPanelSizeX, double dPanelSizeY, double dPixelSize,
			double pdTrans[], double pdTrigs[], 
			unsigned int iNumIllums, unsigned int iTotalNumTrigs, unsigned int iNumCams,
			double dOffsetX, double dOffsetY,
			unsigned int iTileCols, unsigned int iTileRows,
			int iBayerType, unsigned int iFirstPhysicalCam)
	{
		_pAligner->ChangeQXproduction(
		dPanelSizeX, dPanelSizeY, dPixelSize,
		pdTrans, pdTrigs, 
		iNumIllums, iTotalNumTrigs, iNumCams,
		dOffsetX, dOffsetY,
		iTileCols, iTileRows,
		iBayerType, iFirstPhysicalCam);
		return true;
	}

	void ManagedPanelAlignment::SetCoarseConsistCheckToleranceInPixel(double dMaxColInconsistInPixel, double dMaxRowInconsistInPixel)
	{
		_pAligner->SetCoarseConsistCheckToleranceInPixel(dMaxColInconsistInPixel, dMaxRowInconsistInPixel);
	}

	void ManagedPanelAlignment::LogPanelEdgeDebugImages(bool bLog)
	{
		_pAligner->LogPanelEdgeDebugImages(bLog);
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

	void ManagedPanelAlignment::UseCameraModelStitch(bool bValue)
	{
		_pAligner->UseCameraModelStitch(bValue);
	}
	void ManagedPanelAlignment::UseCameraModelIterativeStitch(bool bValue)
	{
		_pAligner->UseCameraModelIterativeStitch(bValue);
	}
	void ManagedPanelAlignment::SetUseTwoPassStitch(bool bValue)
	{
		_pAligner->SetUseTwoPassStitch(bValue);
	}

	void ManagedPanelAlignment::UseProjectiveTransform(bool bValue)
	{
		_pAligner->UseProjectiveTransform(bValue);
	}

	void ManagedPanelAlignment::EnableFiducialAlignmentCheck(bool bValue)
	{
		_pAligner->EnableFiducialAlignmentCheck(bValue);
	}

	void ManagedPanelAlignment::SetPanelEdgeDetection(
		bool bDetectPanelEdge,
		int iLayerIndex4Edge,
		bool bConveyorLeft2Right,
		bool bConveyorFixedFrontRail)
	{
		_pAligner->SetPanelEdgeDetection(
			bDetectPanelEdge, 
			iLayerIndex4Edge,
			bConveyorLeft2Right,
			bConveyorFixedFrontRail);
	}

	void ManagedPanelAlignment::SetCalibrationWeight(double dValue)
	{
		_pAligner->SetCalibrationWeight(dValue);
	}

	void ManagedPanelAlignment::SetSkipDemosaic(bool bValue)
	{
		_pAligner->SetSkipDemosaic(bValue);
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

	bool ManagedPanelAlignment::SaveQXStitchedImage(System::String^ imagePath,int iType)
	{
		System::IntPtr stringPtr = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(imagePath);	

		bool bSaved = _pAligner->SaveQXStitchedImage((char*)(stringPtr.ToPointer()), iType);

		System::Runtime::InteropServices::Marshal::FreeHGlobal(stringPtr);
		return bSaved;
	}

	double ManagedPanelAlignment::GetAlignmentTime() 
	{
		return _pAligner->GetAlignmentTime(); 
	};

	bool ManagedPanelAlignment::GetCamModelPanelHeight(unsigned int iDeviceIndex, array<double>^ pZCoef)
	{
		if(pZCoef->Length < 16)
			return(false);

		double pZ[16];
		if(!_pAligner->GetCamModelPanelHeight(iDeviceIndex, pZ))
			return(false);

		for(int i=0; i<16; i++)
			pZCoef[i] = pZ[i];

		return(true);
	}
}
