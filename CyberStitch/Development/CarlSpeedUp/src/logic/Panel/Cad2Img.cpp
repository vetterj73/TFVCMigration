#include <math.h>
#include "Cad2Img.h"
#include "RenderShape.h"
#include "nint.h"  // Needed for Rudd files

bool Cad2Img::DrawCAD(Panel* pPanel, unsigned char* cadBuffer, bool DrawCADROI)
{
	Image cadImage;
	double resolutionX = pPanel->GetPixelSizeX();
	double resolutionY = pPanel->GetPixelSizeY();
	ImgTransform imgTransform(resolutionX, resolutionY, 0, 0, 0);
	cadImage.Configure(pPanel->GetNumPixelsInY(), pPanel->GetNumPixelsInX(), pPanel->GetNumPixelsInY(), imgTransform, imgTransform, false, cadBuffer);
	
	cadImage.ZeroBuffer();

	for(FeatureListIterator feature = pPanel->beginFeatures(); feature!=pPanel->endFeatures(); ++feature)
	{
		RenderFeature(&cadImage, static_cast<CrossFeature*>(feature->second));
		
		// @todo - This was commented out by Alan to get things to build for cyberstitch...
		// Instead of having the image draw to itself, we should implement a way to draw
		// the rectangular box using a render function.
		//if(drawCADROI)
		//{
		//	Box bounds = (feature->second)->GetBoundingBox();
		//	_cadImage.DrawBox(bounds);
		//	bounds = (feature->second)->GetInspectionArea();
		//	_cadImage.DrawBox(bounds);
		//}
	}

	for(FeatureListIterator feature = pPanel->beginFiducials(); feature!=pPanel->endFiducials(); ++feature)
	{
		RenderFeature(&cadImage, feature->second);
	}

	return true;
}

bool Cad2Img::DrawHeightImage(Panel* pPanel, unsigned char* heightBuffer, double dHeightResolution, double dSlopeInGreyLevel)
{
	Image heightImage;
	double resolutionX = pPanel->GetPixelSizeX();
	double resolutionY = pPanel->GetPixelSizeY();
	ImgTransform imgTransform(resolutionX, resolutionY, 0, 0, 0);
	heightImage.Configure(pPanel->GetNumPixelsInY(), pPanel->GetNumPixelsInX(), pPanel->GetNumPixelsInY(), imgTransform, imgTransform, false, heightBuffer);
	
	heightImage.ZeroBuffer();

	for(FeatureListIterator feature = pPanel->beginFeatures(); feature!=pPanel->endFeatures(); ++feature)
	{
		// only consider rectangle feature
		if(feature->second->GetShape() != Feature::SHAPE_RECTANGLE)
			continue;
		
		// Convert Height to GreyLevel 
		double 	dHeight = ((RectangularFeature*)(feature->second))->GetSizeZ();
		int iGreyLevel = (int)(dHeight/dHeightResolution+0.5);
		if(iGreyLevel <= 0) 
			continue;
		if(iGreyLevel > 255)
			iGreyLevel = 255;
		
		int iAntiAlias = 0;
		if(dSlopeInGreyLevel==0)
			RenderRectangle(heightImage, (RectangularFeature*)feature->second, iGreyLevel, iAntiAlias);
		else
			RenderRectangleForHeight(heightImage, (RectangularFeature*)feature->second, iGreyLevel, dSlopeInGreyLevel);
	}

	/*/int iCount = 0;
	for(int i =0; i<heightImage.BufferSizeInBytes(); i++)
	{
		//if(heightBuffer[i]) iCount++;
		heightBuffer[i] =80;
	}
	//double dRate = (double)iCount /heightImage.BufferSizeInBytes();
	//*/
	//heightImage.Save("C:\\Temp\\Height.bmp");

	return(true);
}

// pPanel: input, panel description
// pMaskBuf: inout, mask image buffer
// iStride: mask image stride
// dMinHeight: only component higher than minimum height will be masked
// dPixelExpansion: pixel expansion from CAD area to create mask
bool Cad2Img::DrawMaskImage(Panel* pPanel, unsigned char* pMaskBuf, int iStride, double dMinHeight, double dPixelExpansion)
{
	Image maskImage;
	double resolutionX = pPanel->GetPixelSizeX();
	double resolutionY = pPanel->GetPixelSizeY();
	ImgTransform imgTransform(resolutionX, resolutionY, 0, 0, 0);
	maskImage.Configure(pPanel->GetNumPixelsInY(), pPanel->GetNumPixelsInX(), iStride, imgTransform, imgTransform, false, pMaskBuf);

	maskImage.ZeroBuffer();

	for(FeatureListIterator feature = pPanel->beginFeatures(); feature!=pPanel->endFeatures(); ++feature)
	{
		// only consider rectangle feature
		if(feature->second->GetShape() != Feature::SHAPE_RECTANGLE)
			continue;
		
		// Min height check
		double 	dHeight = ((RectangularFeature*)(feature->second))->GetSizeZ();
		if(dHeight < dMinHeight)
			continue;

		RectangularFeature* pRect = (RectangularFeature*)feature->second;
		RectangularFeature expRect(0, pRect->GetCadX(), pRect->GetCadY(), pRect->GetRotation(),
			pRect->GetSizeX()+dPixelExpansion*resolutionX, pRect->GetSizeY()+dPixelExpansion*resolutionY, pRect->GetSizeZ());
		
		int iAntiAlias = 0;
		int iGreyLevel = 255;
		RenderRectangle(maskImage, &expRect, iGreyLevel, iAntiAlias);
	}

	// For debug
	// maskImage.Save("C:\\Temp\\Mask.bmp");

	return(true);
}


bool Cad2Img::DrawAperatures(Panel* pPanel, unsigned short* aptBuffer, bool DrawCADROI)
{
	Image16 aptImage;
	double resolutionX = pPanel->GetPixelSizeX();
	double resolutionY = pPanel->GetPixelSizeY();
	ImgTransform imgTransform(resolutionX, resolutionY, 0, 0, 0);
	aptImage.Configure(pPanel->GetNumPixelsInY(), pPanel->GetNumPixelsInX(), pPanel->GetNumPixelsInY(), imgTransform, imgTransform, false, (unsigned char*)aptBuffer);
	
	aptImage.ZeroBuffer();

	for(FeatureListIterator feature = pPanel->beginFeatures(); feature!=pPanel->endFeatures(); ++feature)
	{
		RenderFeature(&aptImage, feature->second);

		// @todo - This was commented out by Alan to get things to build for cyberstitch...
		// Instead of having the image draw to itself, we should implement a way to draw
		// the rectangular box using a render function.
		//if(drawCADROI)
		//{
		//	Box bounds = (feature->second)->GetBoundingBox();
		//	aptImage.DrawBox(bounds);
		//	bounds = (feature->second)->GetInspectionArea();
		//	aptImage.DrawBox(bounds);
		//}

	}
	return true;
}