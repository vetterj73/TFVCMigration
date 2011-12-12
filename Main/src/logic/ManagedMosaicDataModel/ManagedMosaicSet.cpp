#include "stdafx.h"
#include "ManagedMosaicSet.h"

namespace MMosaicDM 
{
	bool ManagedMosaicSet::SaveAllStitchedImagesToDirectory(System::String^ directoryName)
	{
		System::IntPtr stringPtr = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(directoryName);
		std::string nativeDirName = (char*)stringPtr.ToPointer();			

		bool bGood = _pMosaicSet->SaveAllStitchedImagesToDirectory(nativeDirName);
		System::Runtime::InteropServices::Marshal::FreeHGlobal(stringPtr);
		return bGood;
	}

	bool ManagedMosaicSet::LoadAllStitchedImagesFromDirectory(System::String^ directoryName)
	{
		System::IntPtr stringPtr = System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(directoryName);
		std::string nativeDirName = (char*)stringPtr.ToPointer();			

		bool bGood = _pMosaicSet->LoadAllStitchedImagesFromDirectory(nativeDirName);
		System::Runtime::InteropServices::Marshal::FreeHGlobal(stringPtr);
		return bGood;
	}

	bool ManagedMosaicSet::CopyTransforms(ManagedMosaicSet ^pManagedMosaicSet)
	{
		MosaicDM::MosaicSet* pSet =(MosaicDM::MosaicSet*)(void*)pManagedMosaicSet->UnmanagedMosaicSet;
		return _pMosaicSet->CopyTransforms(pSet);
	}

	bool ManagedMosaicSet::CopyBuffers(ManagedMosaicSet ^pManagedMosaicSet)
	{
		MosaicDM::MosaicSet* pSet =(MosaicDM::MosaicSet*)(void*)pManagedMosaicSet->UnmanagedMosaicSet;
		return _pMosaicSet->CopyBuffers(pSet);
	}

	void ManagedMosaicSet::SetComponentHeightInfo(				
		System::IntPtr pHeightBuf,		// Component height image buf
		unsigned int iHeightSpan,		// Component height image span
		double dHeightResolution,		// Height resolution in grey level (meter/grey level)
		double dPupilDistance)			// SIM pupil distance (meter))
	{
		_pMosaicSet->SetComponentHeightInfo(				
		(unsigned char*)(void*)pHeightBuf,		
		iHeightSpan,
		dHeightResolution,	
		dPupilDistance);
	}
}