#include <math.h>
#include "Cad2Img.h"
#include "RenderShape.h"

// Rudd files
#include "nint.h"

Cad2Img::Cad2Img(Panel*			p,
				 unsigned int columns,
				 unsigned int rows,
				 unsigned int stride,
				 unsigned char * cadBuffer,
				 Word * aptBuffer,
				 double	resolution,
				 bool drawCADROI) :
				 _pPanel(p),
				 _resolution(resolution),
				 _drawApt(true),
				 _drawCad(true)
{
	//if(resolution==0.0)
	//	G_LOG_0_ERROR("Cad2Img() - zero pixel size");

	ImgTransform imgTransform(_resolution, _resolution, 0, 0, 0);

	if(cadBuffer)
		_cadImage.Configure(columns, rows, stride, imgTransform, imgTransform, false, cadBuffer);
	else
		_drawCad = false;

	if(aptBuffer)
		_aptImage.Configure(columns, rows, stride, imgTransform, imgTransform, false, (unsigned char*)aptBuffer);
	else
		_drawApt = false;

	DrawPads(drawCADROI);
}

bool Cad2Img::DrawCAD(Panel* pPanel, unsigned char* cadBuffer, bool DrawCADROI)
{
	Image cadImage;
	double resolution = pPanel->GetPixelSizeX();
	ImgTransform imgTransform(resolution, resolution, 0, 0, 0);
	cadImage.Configure(pPanel->GetNumPixelsInY(), pPanel->GetNumPixelsInX(), pPanel->GetNumPixelsInY(), imgTransform, imgTransform, false, cadBuffer);

	for(FeatureListIterator feature = pPanel->beginFeatures(); feature!=pPanel->endFeatures(); ++feature)
	{
		switch(feature->second->GetShape())
		{
			case Feature::SHAPE_CROSS:
				RenderCross(cadImage, resolution, static_cast<CrossFeature*>(feature->second));
				break;
			case Feature::SHAPE_DIAMOND:
				RenderDiamond(cadImage, resolution, static_cast<DiamondFeature*>(feature->second));
				break;
			case Feature::SHAPE_DISC:
				RenderDisc(cadImage, resolution, static_cast<DiscFeature*>(feature->second));
				break;
			case Feature::SHAPE_DONUT:
				RenderDonut(cadImage, resolution, static_cast<DonutFeature*>(feature->second));
				break;
			case Feature::SHAPE_RECTANGLE:
				RenderRectangle(cadImage, resolution, static_cast<RectangularFeature*>(feature->second));
				break;
			case Feature::SHAPE_TRIANGLE:
				RenderTriangle(cadImage, resolution, static_cast<TriangleFeature*>(feature->second));
				break;
			case Feature::SHAPE_CYBER:
				RenderCyberShape(cadImage, resolution, static_cast<CyberFeature*>(feature->second));
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
		//	if(_drawCad)
		//		_cadImage.DrawBox(bounds);
		//	if(_drawApt)
		//		_aptImage.DrawBox(bounds);
		//	bounds = (feature->second)->GetInspectionArea();
		//	if(_drawCad)
		//		_cadImage.DrawBox(bounds);
		//	if(_drawApt)
		//		_aptImage.DrawBox(bounds);
		//}
	}

	for(FeatureListIterator feature = pPanel->beginFiducials(); feature!=pPanel->endFiducials(); ++feature)
	{
		switch((feature->second)->GetShape())
		{
			case Feature::SHAPE_CROSS:
				RenderCross(cadImage, resolution, static_cast<CrossFeature*>(feature->second));
				break;
			case Feature::SHAPE_DIAMOND:
				RenderDiamond(cadImage, resolution, static_cast<DiamondFeature*>(feature->second));
				break;
			case Feature::SHAPE_DISC:
				RenderDisc(cadImage, resolution, static_cast<DiscFeature*>(feature->second));
				break;
			case Feature::SHAPE_DONUT:
				RenderDonut(cadImage, resolution, static_cast<DonutFeature*>(feature->second));
				break;
			case Feature::SHAPE_RECTANGLE:
				RenderRectangle(cadImage, resolution, static_cast<RectangularFeature*>(feature->second));
				break;
			case Feature::SHAPE_TRIANGLE:
				RenderTriangle(cadImage, resolution, static_cast<TriangleFeature*>(feature->second));
				break;
			case Feature::SHAPE_CYBER:
				RenderCyberShape(cadImage, resolution, static_cast<CyberFeature*>(feature->second));
				break;
			case Feature::SHAPE_UNDEFINED:
			default:
				break;  // Do nothing - we don't know how to draw whatever this is...
		}
	}

	return true;
}

void Cad2Img::DrawPads(bool drawCADROI)
{
	for(FeatureListIterator feature = _pPanel->beginFeatures(); feature!=_pPanel->endFeatures(); ++feature)
	{
		switch(feature->second->GetShape())
		{
			case Feature::SHAPE_CROSS:
				if(_drawCad)
					RenderCross(_cadImage, _resolution, static_cast<CrossFeature*>(feature->second));
				if(_drawApt)
					RenderCross(_aptImage, _resolution, static_cast<CrossFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_DIAMOND:
				if(_drawCad)
					RenderDiamond(_cadImage, _resolution, static_cast<DiamondFeature*>(feature->second));
				if(_drawApt)
					RenderDiamond(_aptImage, _resolution, static_cast<DiamondFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_DISC:
				if(_drawCad)
					RenderDisc(_cadImage, _resolution, static_cast<DiscFeature*>(feature->second));
				if(_drawApt)
					RenderDisc(_aptImage, _resolution, static_cast<DiscFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_DONUT:
				if(_drawCad)
					RenderDonut(_cadImage, _resolution, static_cast<DonutFeature*>(feature->second));
				if(_drawApt)
					RenderDonut(_aptImage, _resolution, static_cast<DonutFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_RECTANGLE:
				if(_drawCad)
					RenderRectangle(_cadImage, _resolution, static_cast<RectangularFeature*>(feature->second));
				if(_drawApt)
					RenderRectangle(_aptImage, _resolution, static_cast<RectangularFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_TRIANGLE:
				if(_drawCad)
					RenderTriangle(_cadImage, _resolution, static_cast<TriangleFeature*>(feature->second));
				if(_drawApt)
					RenderTriangle(_aptImage, _resolution, static_cast<TriangleFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
				break;
			case Feature::SHAPE_CYBER:
				if(_drawCad)
					RenderCyberShape(_cadImage, _resolution, static_cast<CyberFeature*>(feature->second));
				if(_drawApt)
					RenderCyberShape(_aptImage, _resolution, static_cast<CyberFeature*>(feature->second), (feature->second)->GetApertureValue(), 0);
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
		//	if(_drawCad)
		//		_cadImage.DrawBox(bounds);
		//	if(_drawApt)
		//		_aptImage.DrawBox(bounds);
		//	bounds = (feature->second)->GetInspectionArea();
		//	if(_drawCad)
		//		_cadImage.DrawBox(bounds);
		//	if(_drawApt)
		//		_aptImage.DrawBox(bounds);
		//}
	}

	if(_drawCad)  // Only used for CAD - not SPI Aperatures...
	{
		for(FeatureListIterator feature = _pPanel->beginFiducials(); feature!=_pPanel->endFiducials(); ++feature)
		{
			switch((feature->second)->GetShape())
			{
				case Feature::SHAPE_CROSS:
					RenderCross(_cadImage, _resolution, static_cast<CrossFeature*>(feature->second));
					break;
				case Feature::SHAPE_DIAMOND:
					RenderDiamond(_cadImage, _resolution, static_cast<DiamondFeature*>(feature->second));
					break;
				case Feature::SHAPE_DISC:
					RenderDisc(_cadImage, _resolution, static_cast<DiscFeature*>(feature->second));
					break;
				case Feature::SHAPE_DONUT:
					RenderDonut(_cadImage, _resolution, static_cast<DonutFeature*>(feature->second));
					break;
				case Feature::SHAPE_RECTANGLE:
					RenderRectangle(_cadImage, _resolution, static_cast<RectangularFeature*>(feature->second));
					break;
				case Feature::SHAPE_TRIANGLE:
					RenderTriangle(_cadImage, _resolution, static_cast<TriangleFeature*>(feature->second));
					break;
				case Feature::SHAPE_CYBER:
					RenderCyberShape(_cadImage, _resolution, static_cast<CyberFeature*>(feature->second));
					break;
				case Feature::SHAPE_UNDEFINED:
				default:
					break;  // Do nothing - we don't know how to draw whatever this is...
			}
		}
	}
}