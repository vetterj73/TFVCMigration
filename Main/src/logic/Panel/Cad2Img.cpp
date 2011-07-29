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

	for(FeatureListIterator feature = pPanel->beginFeatures(); feature!=pPanel->endFeatures(); ++feature)
	{
		RenderFeature(&cadImage, resolutionX, static_cast<CrossFeature*>(feature->second));
		
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
		RenderFeature(&cadImage, resolutionX, feature->second);
	}

	return true;
}

bool Cad2Img::DrawAperatures(Panel* pPanel, unsigned short* aptBuffer, bool DrawCADROI)
{
	Image16 aptImage;
	double resolutionX = pPanel->GetPixelSizeX();
	double resolutionY = pPanel->GetPixelSizeY();
	ImgTransform imgTransform(resolutionX, resolutionY, 0, 0, 0);
	aptImage.Configure(pPanel->GetNumPixelsInY(), pPanel->GetNumPixelsInX(), pPanel->GetNumPixelsInY(), imgTransform, imgTransform, false, (unsigned char*)aptBuffer);
	
	for(FeatureListIterator feature = pPanel->beginFeatures(); feature!=pPanel->endFeatures(); ++feature)
	{
		switch(feature->second->GetShape())
		{
			RenderFeature(&aptImage, resolutionX, feature->second);
		}

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