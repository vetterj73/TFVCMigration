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

	bool ManagedMosaicSet::AddSubDeviceInfo(unsigned int iDeviceIndex, array<unsigned int>^ pLastCameras)
	{
		if(pLastCameras->Length <= 1)
			return(false);

		list<unsigned int> tempList;
		for(int i=0; i<pLastCameras->Length; i++)
			tempList.push_back(pLastCameras[i]);

		_pMosaicSet->AddSubDeviceInfo(iDeviceIndex, tempList);

		return(true);
	}
}