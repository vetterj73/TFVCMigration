#include "StdAfx.h"
#include "ManagedFeatureLocationCheck.h"
#include "Image.h"

namespace PanelAlignM {

	ManagedFeatureLocationCheck::ManagedFeatureLocationCheck(CPanel^ panel)
	{
		_pPanel  = (Panel*)(void*)panel->UnmanagedPanel;
		_pChecker = new FeatureLocationCheck(_pPanel);
	}

	bool ManagedFeatureLocationCheck::CheckFeatureLocation(System::IntPtr pData, array<double>^ pdResults)
	{
		// Create image for process
		ImgTransform inputTransform;
		inputTransform.Config(_pPanel->GetPixelSizeX(), 
				_pPanel->GetPixelSizeX(), 0, 0, 0);

		Image image
			(_pPanel->GetNumPixelsInY(),	// Columns
			_pPanel->GetNumPixelsInX(),		// Rows	
			_pPanel->GetNumPixelsInY(),		// In pixels
			1,								// Bytes per pixel
			inputTransform,					
			inputTransform,		
			false,							// Falg for whether create own buffer
			(unsigned char*)(void*)pData);

		// Fiducial check
		int iNum = pdResults->Length;
		double* pd = new double[iNum];
		_pChecker->CheckFeatureLocation(&image, pd);
		for(int i=0; i<iNum; i++)
		{
			pdResults[i] = pd[i];
		}

		delete [] pd;

		return true;
	}

}