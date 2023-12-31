#include <math.h>
#include "Cad2Img.h"
#include "RenderShape.h"

// Rudd files
#include "nint.h"

Cad2Img::Cad2Img(Panel*			p,
				 unsigned int columns,
				 unsigned int rows,
				 unsigned int stride,
				 Byte * cadBuffer,
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

	//if(_cadImage.GetBuffer()!=0)
	//	G_LOG_0_ERROR("Cad2Img() - cannot overwrite CadImg");

	//if(_aptImage.GetBuffer()!=0)
	//	G_LOG_0_ERROR("Cad2Img() - cannot overwrite CadImg");

	ImgTransform imgTransform(_resolution, _resolution, 0, 0, 0);

	if(cadBuffer)
		_cadImage.Configure(columns, rows, stride, imgTransform, imgTransform, false, cadBuffer);
	else
		_drawCad = false;

	if(aptBuffer)
		_aptImage.Configure(columns, rows, stride, imgTransform, imgTransform, false, (Byte*)aptBuffer);
	else
		_drawApt = false;

	DrawPads(drawCADROI);
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
		//		G_LOG_0_ERROR("DrawPads() - undefined shape");
				break;

			default:
		//		G_LOG_0_ERROR("DrawPads() - undefined default shape");
				break;
		}

		// @todo - This was commented out by Alan to get things to build for cyberstitch...
		// Do we need it?
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

	if(0)//_drawCad)
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
	//				G_LOG_0_ERROR("DrawPads() - undefined shape");
					break;

				default:
	//				G_LOG_0_ERROR("DrawPads() - undefined default shape");
					break;
			}
		}
	}
}