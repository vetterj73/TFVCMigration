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
		switch(feature->second->GetShape())
		{
			case Feature::SHAPE_CROSS:
				RenderCross(cadImage, resolutionX, static_cast<CrossFeature*>(feature->second));
				break;
			case Feature::SHAPE_DIAMOND:
				RenderDiamond(cadImage, resolutionX, static_cast<DiamondFeature*>(feature->second));
				break;
			case Feature::SHAPE_DISC:
				RenderDisc(cadImage, resolutionX, static_cast<DiscFeature*>(feature->second));
				break;
			case Feature::SHAPE_DONUT:
				RenderDonut(cadImage, resolutionX, static_cast<DonutFeature*>(feature->second));
				break;
			case Feature::SHAPE_RECTANGLE:
				RenderRectangle(cadImage, resolutionX, static_cast<RectangularFeature*>(feature->second));
				break;
			case Feature::SHAPE_TRIANGLE:
				RenderTriangle(cadImage, resolutionX, static_cast<TriangleFeature*>(feature->second));
				break;
			case Feature::SHAPE_CYBER:
				RenderCyberShape(cadImage, resolutionX, static_cast<CyberFeature*>(feature->second));
				break;
			case Feature::SHAPE_UNDEFINED:
			default:  // Do nothing - we don't know how to draw whatever this is...
				break;
		}
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
		switch((feature->second)->GetShape())
		{
			case Feature::SHAPE_CROSS:
				RenderCross(cadImage, resolutionX, static_cast<CrossFeature*>(feature->second));
				break;
			case Feature::SHAPE_DIAMOND:
				RenderDiamond(cadImage, resolutionX, static_cast<DiamondFeature*>(feature->second));
				break;
			case Feature::SHAPE_DISC:
				RenderDisc(cadImage, resolutionX, static_cast<DiscFeature*>(feature->second));
				break;
			case Feature::SHAPE_DONUT:
				RenderDonut(cadImage, resolutionX, static_cast<DonutFeature*>(feature->second));
				break;
			case Feature::SHAPE_RECTANGLE:
				RenderRectangle(cadImage, resolutionX, static_cast<RectangularFeature*>(feature->second));
				break;
			case Feature::SHAPE_TRIANGLE:
				RenderTriangle(cadImage, resolutionX, static_cast<TriangleFeature*>(feature->second));
				break;
			case Feature::SHAPE_CYBER:
				RenderCyberShape(cadImage, resolutionX, static_cast<CyberFeature*>(feature->second));
				break;
			case Feature::SHAPE_UNDEFINED:
			default:
				break;  // Do nothing - we don't know how to draw whatever this is...
		}
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
			case Feature::SHAPE_CROSS:
				RenderCross(aptImage, resolutionX, static_cast<CrossFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_DIAMOND:
				RenderDiamond(aptImage, resolutionX, static_cast<DiamondFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_DISC:
				RenderDisc(aptImage, resolutionX, static_cast<DiscFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_DONUT:
				RenderDonut(aptImage, resolutionX, static_cast<DonutFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_RECTANGLE:
				RenderRectangle(aptImage, resolutionX, static_cast<RectangularFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_TRIANGLE:
				RenderTriangle(aptImage, resolutionX, static_cast<TriangleFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_CYBER:
				RenderCyberShape(aptImage, resolutionX, static_cast<CyberFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_UNDEFINED:
			default:  // Do nothing - we don't know how to draw whatever this is...
				break;
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