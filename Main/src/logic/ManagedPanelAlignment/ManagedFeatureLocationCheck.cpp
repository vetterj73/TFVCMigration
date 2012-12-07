#include "StdAfx.h"
#include "ManagedFeatureLocationCheck.h"
#include "Image.h"

namespace PanelAlignM {

	ManagedFeatureLocationCheck::ManagedFeatureLocationCheck(CPanel^ panel)
	{
		_pPanel  = (Panel*)(void*)panel->UnmanagedPanel;
		_pChecker = new FeatureLocationCheck(_pPanel);
	}

	bool ManagedFeatureLocationCheck::CheckFeatureLocation(System::IntPtr pData, int iSpan, array<double>^ pdResults)
	{
		// Create image for process
		ImgTransform inputTransform;
		inputTransform.Config(_pPanel->GetPixelSizeX(), 
				_pPanel->GetPixelSizeX(), 0, 0, 0);

		Image image
			(_pPanel->GetNumPixelsInY(),	// Columns
			_pPanel->GetNumPixelsInX(),		// Rows	
			iSpan,							// In pixels
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

	ManagedImageFidAligner::ManagedImageFidAligner(CPanel^ panel)
	{
		_pPanel  = (Panel*)(void*)panel->UnmanagedPanel;
		_imageFidAligner = new ImageFidAligner(_pPanel);
	}

	bool ManagedImageFidAligner::CalculateTransform(System::IntPtr pData, int iSpan, array<double>^ zCof, array<double>^ trans)
	{
		// Create image for process
		ImgTransform inputTransform;
		inputTransform.Config(_pPanel->GetPixelSizeX(), 
				_pPanel->GetPixelSizeX(), 0, 0, 0);

		Image image
			(_pPanel->GetNumPixelsInY(),	// Columns
			_pPanel->GetNumPixelsInX(),		// Rows	
			iSpan,							// In pixels
			1,								// Bytes per pixel
			inputTransform,					
			inputTransform,		
			false,							// Falg for whether create own buffer
			(unsigned char*)(void*)pData);
		
		double* pZ = NULL;
		if(zCof != nullptr)
		{
			pZ = new double[16];
			for(int i=0; i<16; i++)
				pZ[i] = zCof[i];
		}
		double t[3][3];
		if(!_imageFidAligner->CalculateTransform(&image, t, pZ))
			return(false);

		for(int i=0; i<9; i++)
			trans[i] = t[i/3][i%3];

		if(pZ!=NULL)
			delete [] pZ;

		return(true);
	}

	System::IntPtr ManagedImageFidAligner::MorphImage(System::IntPtr pDataIn, int iSpanIn,
		array<double>^ zCof)
	{
		// Create image for process
		ImgTransform inputTransform;
		inputTransform.Config(_pPanel->GetPixelSizeX(), 
				_pPanel->GetPixelSizeX(), 0, 0, 0);

		Image image
			(_pPanel->GetNumPixelsInY(),	// Columns
			_pPanel->GetNumPixelsInX(),		// Rows	
			iSpanIn,							// In pixels
			1,								// Bytes per pixel
			inputTransform,					
			inputTransform,		
			false,							// Falg for whether create own buffer
			(unsigned char*)(void*)pDataIn);

		// transfer Z
		double* pZ = NULL;
		if(zCof != nullptr)
		{
			pZ = new double[16];
			for(int i=0; i<16; i++)
				pZ[i] = zCof[i];
		}
		Image* pImgOut = _imageFidAligner->MorphImage(&image, pZ);

		if(pZ!=NULL)
			delete [] pZ;

		 return (System::IntPtr)(pImgOut->GetBuffer());
	}
}