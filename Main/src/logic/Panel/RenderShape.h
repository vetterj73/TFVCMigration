/*

	Renders shapes for use as masks

*/

#pragma once

#include "Image.h"
#include "Image16.h"
#include "Feature.h"

// The following rountines use Rudd's ScanConv implemented in RenderShape
template <typename IMAGETYPE>
void RenderShape(IMAGETYPE& image, double resolution, 
				 unsigned int featGrayValue, int featType, int antiAlias,
				 double height, double width, double rotation, double radius, 
				 double xCenter, double yCenter);

template<typename IMAGETYPE>
void RenderDonut(IMAGETYPE& image, double resolution, DonutFeature* donut, unsigned int grayValue=255, int antiAlias=1);

template<typename IMAGETYPE>
void RenderDisc(IMAGETYPE& image, double resolution, DiscFeature* disc, unsigned int grayValue=255, int antiAlias=1);

// The following routines use Rudd's aapoly to draw polygons implemented in RenderPolygon
template<typename IMAGETYPE>
void RenderAAPolygon(IMAGETYPE& image, Feature* feature, PointList& polygonPoints);

template<typename IMAGETYPE>
void RenderPolygon(IMAGETYPE& image, double resolution, 
				   Feature* feature, PointList polygonPoints[], int numPolygons, 
				   unsigned int grayValue, int antiAlias);

template<typename IMAGETYPE>
void RenderCyberShape(IMAGETYPE& image, double resolution, CyberFeature* cyberShape, unsigned int grayValue=255, int antiAlias=1);

template<typename IMAGETYPE>
void RenderCross(IMAGETYPE& image, double resolution, CrossFeature* cross, unsigned int grayValue=255, int antiAlias=1);

template<typename IMAGETYPE>
void RenderDiamond(IMAGETYPE& image, double resolution, DiamondFeature* diamond, unsigned int grayValue=255, int antiAlias=1);

template <typename IMAGETYPE>
void RenderDiamondFrame(IMAGETYPE& image, double resolution, DiamondFrameFeature* diamondFrame, unsigned int grayValue=255, int antiAlias=1);

template<typename IMAGETYPE>
void RenderRectangle(IMAGETYPE& image, double resolution, RectangularFeature* rect, unsigned int grayValue=255, int antiAlias=1);

template <typename IMAGETYPE>
void RenderRectangleFrame(IMAGETYPE& image, double resolution, RectangularFrameFeature* rectFrame, unsigned int grayValue=255, int antiAlias=1);

template<typename IMAGETYPE>
void RenderTriangle(IMAGETYPE& image, double resolution, TriangleFeature* triangle, unsigned int grayValue=255, int antiAlias=1);

template <typename IMAGETYPE>
void RenderTriangleFrame(IMAGETYPE& image, double resolution, EquilateralTriangleFrameFeature* triangleFrame, unsigned int grayValue=255, int antiAlias=1);

template <typename IMAGETYPE>
void RenderCheckerPattern(IMAGETYPE& image, double resolution, CheckerPatternFeature* checkerPattern, unsigned int grayValue=255, int antiAlias=1);

template <typename IMAGETYPE>
bool RenderFeature(IMAGETYPE* pImg, double dResolution, Feature* pFeature, unsigned int grayValue=255, int antiAlias=1);