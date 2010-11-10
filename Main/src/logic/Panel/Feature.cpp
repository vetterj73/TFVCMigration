#include "ArcPolygonizer.h"
#include "Feature.h"
#include "Panel.h"

#ifndef EPSILON
#define EPSILON 0.0000001
#endif

#ifndef PI
#define PI 3.14159265358979323846
#endif

//////////////////////////////////////////////////////////////////////////
////////////////////////                      ////////////////////////////
////////////////////////  Feature Base Class  ////////////////////////////
////////////////////////                      ////////////////////////////
//////////////////////////////////////////////////////////////////////////
Feature::Feature() : _status(FEATURE_UNINITIALIZED),
				_shape(SHAPE_UNDEFINED),
				_name(""),
				_index(0),
				_xCadCenter(0),
				_yCadCenter(0),
				_rotation(0),
				_apertureValue(0),
				_nominalArea(0),
				_percentagePadCoverage(0),
				_xPasteCenter(0),
				_yPasteCenter(0),
				_bridgeWidth(0)
{		
		
}

Feature::Feature(PadShape shape, int id, double positionX, double positionY, double rotation) :
				_status(FEATURE_UNINITIALIZED),
				_shape(shape),
				_name(""),
				_index(id),
				_xCadCenter(positionX),
				_yCadCenter(positionY),
				_rotation(rotation),
				_apertureValue(0),
				_nominalArea(0),
				_percentagePadCoverage(0),
				_xPasteCenter(0),
				_yPasteCenter(0),
				_bridgeWidth(0)
{
}

Feature::~Feature()
{

}

void Feature::SetResults(double coveragePercent, double bridgeWidth, double xCenter, double yCenter)
{
	_percentagePadCoverage = coveragePercent;
	_xPasteCenter = xCenter;
	_yPasteCenter =yCenter;
	_bridgeWidth = bridgeWidth;
	SetStatus(Feature::FEATURE_INSPECTED);
}

void Feature::ResetResults()
{
	_percentagePadCoverage = 0.0;
	_xPasteCenter = 0.0;
	_yPasteCenter = 0.0;
	_bridgeWidth = 0.0;
	SetStatus(Feature::FEATURE_UNINSPECTED);
}


void Feature::InspectionAreaFromBounds()
{
	// @todo - config?
	// Read percent of feature to add to each side from config, different amount based on dimension
	double iaLongDim = Panel::_padInspectionAreaLong;
	double iaShortDim = Panel::_padInspectionAreaShort;

	bool square = _boundingBox.Square();
	double height = _boundingBox.Height();
	double width = _boundingBox.Width();

	double paddingX = 0.0;
	double paddingY = 0.0;

	if(square)
	{
		paddingX = paddingY = height * iaShortDim;
	}
	else if( width > height)
	{
		// wider than tall
		paddingX = width * iaLongDim;
		paddingY = height * iaShortDim;
	}
	else
	{
		// taller than wide
		paddingY = height * iaLongDim;
		paddingX = width * iaShortDim;
	}
	
	_inspectionArea.p1.x = _boundingBox.p1.x - paddingX;
	_inspectionArea.p1.y = _boundingBox.p1.y - paddingY;
	_inspectionArea.p2.x = _boundingBox.p2.x + paddingX;
	_inspectionArea.p2.y = _boundingBox.p2.y + paddingY;
}

bool Feature::Validate(double panelSizeX, double panelSizeY)
{
	// Make sure feature is inside the panel
	if(_xCadCenter<0 || _xCadCenter>panelSizeX)
		return false;

	if(_yCadCenter<0 || _yCadCenter>panelSizeY)
		return false;


	//
	// Make sure the entire feature is on the panel
	if(_boundingBox==0)
		return false;

	Point minPoint = _boundingBox.Min();
	Point maxPoint = _boundingBox.Max();

	if(minPoint.x<0 || minPoint.y<0)
		return false;

	if(maxPoint.x>panelSizeX || maxPoint.y>panelSizeY)
		return false;


	//
	// Check that the inspection area has been calculated.
	if(_inspectionArea==0)
		return false;

	if(_inspectionArea.p1.x < EPSILON)
		_inspectionArea.p1.x = 0.0;
	
	if(_inspectionArea.p1.y < EPSILON)
		_inspectionArea.p1.y = 0.0;

	if(_inspectionArea.p2.x-panelSizeX > EPSILON)
		_inspectionArea.p2.x = panelSizeX;

	if(_inspectionArea.p2.y-panelSizeY > EPSILON)
		_inspectionArea.p2.y = panelSizeY;

	_status = Feature::FEATURE_UNINSPECTED;

	return true;
}



///////////////////////////////////////////////////////////////////////////
////////////////                                     //////////////////////
//////////////// CrossFeature Derived Class          //////////////////////
////////////////                                     //////////////////////
///////////////////////////////////////////////////////////////////////////

CrossFeature::CrossFeature(int id, double positionX, double positionY, double rotation,
						   double sizeX, double sizeY, double legSizeX, double legSizeY) :
							Feature(SHAPE_CROSS, id, positionX, positionY, rotation),
							_sizeX(sizeX),
							_sizeY(sizeY),
							_legSizeX(legSizeX),
							_legSizeY(legSizeY)
{
	_polygonPoints.clear();

	Bound();
	NominalArea();
	InspectionArea();
}


CrossFeature::~CrossFeature()
{
	Feature::~Feature();
	
}

void CrossFeature::Bound()
{
	double radians = _rotation * PI / 180.0;
	bool rotated = (fabs(radians) > 0.0001);

	Point point;

	
	//         8*---* 1
	//          |   |
	//          |   |
	//          |   |
	// 7 *------*   *------* 2
	//   |                 |
	// 6 *------*   *------* 3
	//          |   |
	//          |   |
	//          |   |
	//        5 *---* 4
	
	// Point 1
	point.x = +_legSizeX / 2.0;
	point.y = +_sizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 2
	point.x = +_sizeX / 2.0;
	point.y = +_legSizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 3
	point.x = +_sizeX / 2.0;
	point.y = -_legSizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 4
	point.x = +_legSizeX / 2.0;
	point.y = -_sizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 5
	point.x = -_legSizeX / 2.0;
	point.y = -_sizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 6
	point.x = -_sizeX / 2.0;
	point.y = -_legSizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 7
	point.x = -_sizeX / 2.0;
	point.y = +_legSizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 8
	point.x = -_legSizeX / 2.0;
	point.y = +_sizeY / 2.0;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	Box b(Point(+1E+10,+1E+10), Point(-1E+10,-1E+10));

	PointList::iterator p=_polygonPoints.begin();
	for(p; p!=_polygonPoints.end(); p++)
	{
		if(p->x<b.p1.x)
			b.p1.x=p->x;
		if(p->y<b.p1.y)
			b.p1.y=p->y;

		if(p->x>b.p2.x)
			b.p2.x=p->x;
		if(p->y>b.p2.y)
			b.p2.y=p->y;
	}

	_boundingBox = b;
}


void CrossFeature::NominalArea()
{
	double intersectionArea = _legSizeX * _legSizeY;
	double horizontalBarArea = _legSizeY * _sizeX;
	double vertcalBarArea = _legSizeX * _sizeY;

	_nominalArea = horizontalBarArea + vertcalBarArea - intersectionArea;
}

void CrossFeature::InspectionArea()
{
	if(_boundingBox==0)
		Bound();

	InspectionAreaFromBounds();
}

///////////////////////////////////////////////////////////////////////////
////////////////                                     //////////////////////
//////////////// DiamondFeature Derived Class        //////////////////////
////////////////                                     //////////////////////
///////////////////////////////////////////////////////////////////////////

DiamondFeature::DiamondFeature(int id, double positionX, double positionY, double rotation,
							   double sizeX, double sizeY ) :
								Feature(SHAPE_DIAMOND, id, positionX, positionY, rotation),
								_sizeX(sizeX),
								_sizeY(sizeY)		
{
	_polygonPoints.clear();

	Bound();
	NominalArea();
	InspectionArea();
}

DiamondFeature::~DiamondFeature()
{
	Feature::~Feature();
	
}

void DiamondFeature::Bound()
{
	Point point;
	double x, y, base, height;
	double radians = _rotation * PI / 180.0;
	bool rotated = (fabs(radians)>0.0001);

	x = _xCadCenter;
	y = _yCadCenter;
	base = _sizeX;
	height = _sizeY;


	// Polygon points in CW
	//        * 1
	//       / \
	//      /   \
	//   4 *     * 2
	//      \   /
	//       \ /
	//        * 3

	// Point 1
	point.x = 0.0;
	point.y = height / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	_polygonPoints.push_back(point);

	// Point 2
	point.x = base / 2.0;
	point.y = 0.0;
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	_polygonPoints.push_back(point);

	// Point 3
	point.x = 0.0;
	point.y = -height / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	_polygonPoints.push_back(point);

	// Point 4
	point.x = -base / 2.0;
	point.y = 0.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	_polygonPoints.push_back(point);


	//
	// Calculate bounding box
	Box b(Point(+1E+10,+1E+10), Point(-1E+10,-1E+10));

	PointList::iterator p=_polygonPoints.begin();
	for(p; p!=_polygonPoints.end(); p++)
	{
		if(p->x<b.p1.x)
			b.p1.x=p->x;
		if(p->y<b.p1.y)
			b.p1.y=p->y;

		if(p->x>b.p2.x)
			b.p2.x=p->x;
		if(p->y>b.p2.y)
			b.p2.y=p->y;
	}

	_boundingBox = b;
}

void DiamondFeature::InspectionArea()
{
	if(_boundingBox==0)
		Bound();

	InspectionAreaFromBounds();
}

void DiamondFeature::NominalArea()
{
	_nominalArea = _sizeX * _sizeY;
}

//////////////////////////////////////////////////////////////////////
/////////////////                               //////////////////////
/////////////////  DiscFeature Derived Class    //////////////////////
/////////////////                               //////////////////////
//////////////////////////////////////////////////////////////////////

DiscFeature::DiscFeature(int id, double positionX, double positionY, double diameter) :
						Feature(SHAPE_DISC, id, positionX, positionY, 0),
						_diameter(diameter)
						
{
	Bound();
	NominalArea();
	InspectionArea();
}

DiscFeature::~DiscFeature()
{
	Feature::~Feature();
}

void DiscFeature::Bound()
{
	double radius = _diameter / 2.0;
	
	_boundingBox.p1.x = _xCadCenter - radius;
	_boundingBox.p1.y = _yCadCenter - radius;
	_boundingBox.p2.x = _xCadCenter + radius;
	_boundingBox.p2.y = _yCadCenter + radius;
}

void DiscFeature::InspectionArea()
{
	if(_boundingBox==0)
		Bound();

	InspectionAreaFromBounds();
}

void DiscFeature::NominalArea()
{
	_nominalArea = PI * _diameter * _diameter / 4.0;
}


///////////////////////////////////////////////////////////////////////////
////////////////                                     //////////////////////
//////////////// DonutFeature Derived Class          //////////////////////
////////////////                                     //////////////////////
///////////////////////////////////////////////////////////////////////////

DonutFeature::DonutFeature(int id, double positionX, double positionY,
						   double diameterInside, double diameterOutside) :
						Feature(SHAPE_DONUT, id, positionX, positionY, 0),
						_diameterInside(diameterInside),
						_diameterOutside(diameterOutside)
						
{
	Bound();
	NominalArea();
	InspectionArea();
}

DonutFeature::~DonutFeature()
{
	Feature::~Feature();
	
}

void DonutFeature::Bound()
{
	double radius = _diameterOutside / 2.0;
	
	_boundingBox.p1.x = _xCadCenter - radius;
	_boundingBox.p1.y = _yCadCenter - radius;
	_boundingBox.p2.x = _xCadCenter + radius;
	_boundingBox.p2.y = _yCadCenter + radius;
}

void DonutFeature::InspectionArea()
{
	if(_boundingBox==0)
		Bound();

	InspectionAreaFromBounds();
}

void DonutFeature::NominalArea()
{
	double innerArea = PI * _diameterInside * _diameterInside / 4.0;
	double outerArea = PI * _diameterOutside * _diameterOutside / 4.0;

	_nominalArea = outerArea - innerArea;
}


///////////////////////////////////////////////////////////////////////////
////////////////                                     //////////////////////
/////////////////  RectangularFeature Derived Class  //////////////////////
////////////////                                     //////////////////////
///////////////////////////////////////////////////////////////////////////

RectangularFeature::RectangularFeature(int id, double positionX, double positionY, double rotation,
									   double sizeX, double sizeY ):
									Feature(SHAPE_RECTANGLE, id, positionX, positionY, rotation),
									_width(sizeX),
									_height(sizeY)					
{
	_polygonPoints.clear();

	Bound();
	NominalArea();
	InspectionArea();
}

RectangularFeature::~RectangularFeature()
{
	Feature::~Feature();
	
}

void RectangularFeature::Bound()
{
	double halfHeight = _height / 2.0;
	double halfWidth = _width / 2.0;
	double radians = _rotation * PI / 180.0;
	bool rotated = (fabs(radians)>0.0001);

	Point point;

	
	// 4 *--------* 1
	//   |        |
	//   |        |
	//   |        |
	// 3 *--------* 2
	
	// Point 1
	point.x = +halfWidth;
	point.y = +halfHeight;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 2
	point.x = +halfWidth;
	point.y = -halfHeight;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 3
	point.x = -halfWidth;
	point.y = -halfHeight;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	// Point 4
	point.x = -halfWidth;
	point.y = +halfHeight;
	if(rotated) point.rot(radians);
	point.x += _xCadCenter;
	point.y += _yCadCenter;
	_polygonPoints.push_back(point);

	Box b(Point(+1E+10,+1E+10), Point(-1E+10,-1E+10));

	PointList::iterator p=_polygonPoints.begin();
	for(p; p!=_polygonPoints.end(); p++)
	{
		if(p->x<b.p1.x)
			b.p1.x=p->x;
		if(p->y<b.p1.y)
			b.p1.y=p->y;

		if(p->x>b.p2.x)
			b.p2.x=p->x;
		if(p->y>b.p2.y)
			b.p2.y=p->y;
	}

	_boundingBox = b;
}

void RectangularFeature::InspectionArea()
{
	if(_boundingBox==0)
		Bound();

	InspectionAreaFromBounds();
}

void RectangularFeature::NominalArea()
{
	_nominalArea = _width * _height;
}



///////////////////////////////////////////////////////////////////////////
////////////////                                     //////////////////////
//////////////// TriangleFeature Derived Class       //////////////////////
////////////////                                     //////////////////////
///////////////////////////////////////////////////////////////////////////

TriangleFeature::TriangleFeature(int id, double positionX, double positionY, double rotation,
								 double sizeX, double sizeY, double offsetX ) :
							Feature(SHAPE_TRIANGLE, id, positionX, positionY, rotation),
							_sizeX(sizeX),
							_sizeY(sizeY),
							_offset(offsetX)
						
{
	_polygonPoints.clear();

	Bound();
	NominalArea();
	InspectionArea();
}

TriangleFeature::~TriangleFeature()
{
	Feature::~Feature();
	
}


void TriangleFeature::Bound()
{
	Point point;
	double x, y, base, height, offset;
	double radians = _rotation * PI / 180.0;
	bool rotated = (fabs(radians)>0.0001);

	x = _xCadCenter;
	y = _yCadCenter;
	base = _sizeX;
	height = _sizeY;
	offset = _offset;

	// CAD polygon points in CW order
	//
	//           |---| Offset
	//           |   * 2---------      
	//           |  / \       |
	//           | / * \    height
	//           |/     \     |
	// (x0,y0) 1 *-------* 3-----
	//           |-base--|
	
	double x0 = -(base + offset) / 3.0;
	double y0 = -(height / 3.0);

	// Point 1
	point.x = x0;
	point.y = y0;
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	_polygonPoints.push_back(point);

	// Point 2
	point.x = x0 + offset;
	point.y = y0 + height; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	_polygonPoints.push_back(point);

	// Point 3
	point.x = x0 + base;
	point.y = y0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	_polygonPoints.push_back(point);

	// 
	// Find bounding box
	Box b(Point(+1E+10,+1E+10), Point(-1E+10,-1E+10));

	PointList::iterator p=_polygonPoints.begin();
	for(p; p!=_polygonPoints.end(); p++)
	{
		if(p->x<b.p1.x)
			b.p1.x=p->x;
		if(p->y<b.p1.y)
			b.p1.y=p->y;

		if(p->x>b.p2.x)
			b.p2.x=p->x;
		if(p->y>b.p2.y)
			b.p2.y=p->y;
	}

	_boundingBox = b;
}

void TriangleFeature::InspectionArea()
{
	if(_boundingBox==0)
		Bound();

	InspectionAreaFromBounds();
}

void TriangleFeature::NominalArea()
{
	_nominalArea = _sizeX * _sizeY / 2.0;
}


///////////////////////////////////////////////////////////////////////////
////////////////                                     //////////////////////
//////////////// CyberSegment Class                  //////////////////////
////////////////                                     //////////////////////
///////////////////////////////////////////////////////////////////////////
CyberSegment::CyberSegment(bool line, bool penDown, bool clockwiseArc, double positionX,
				 double positionY, double arcX, double arcY)
{
	_line = line;
	_penDown = penDown;
	_clockwiseArc = clockwiseArc;
	_positionX = positionX;
	_positionY = positionY;
	_arcX = arcX;
	_arcY = arcY;
}


///////////////////////////////////////////////////////////////////////////
////////////////                                     //////////////////////
//////////////// CyberFeature Derived Class          //////////////////////
////////////////                                     //////////////////////
///////////////////////////////////////////////////////////////////////////

CyberFeature::CyberFeature(int id, double positionX, double positionY, double rotation) :
						Feature(SHAPE_CYBER, id, positionX, positionY, rotation),
						_concavePolygon(false)					
{
	_currentSegment = _segments.end();	
}

CyberFeature::~CyberFeature()
{
	for(unsigned int i=0; i<_segments.size(); i++)
		delete _segments[i];

	_segments.clear();
}

void CyberFeature::AddSegment(CyberSegment *segment)
{
	_segments.push_back(segment);
}

CyberSegment* CyberFeature::GetFirstSegment()
{
	_currentSegment = _segments.begin();
	return GetNextSegment();
}

CyberSegment* CyberFeature::GetNextSegment()
{
	CyberSegment* pSegment = NULL;

	if(_currentSegment != _segments.end())
	{
		pSegment = (*_currentSegment);
		_currentSegment++;
	}

	return pSegment;
}

int CyberFeature::GetNumberSegments()
{
	return (int) _segments.size();
}

bool CyberFeature::IsConcave()
{
	return _concavePolygon;
}

bool CyberFeature::Validate(double panelSizeX, double panelSizeY)
{
	//
	// Check if the segments define a valid polygon
	//
	
	PointList segmentVertices; // must be in meters for ArcPolygonizer

	_concavePolygon = false;
	_polygonPoints.clear();

	//
	// Step 1: Convert lines and arcs defined by 
	//         CyberSegments to a series of CW points
	//
	if(_segments.size() <= 0)
	{

#pragma warning("Logging");
//		G_LOG_1_ERROR("CyberShape %d is invalid! There are no segments!", _index);
		return false;
	}

	// Get first segment
	CyberSegment *seg = _segments[0];

	// The first segment must not be an arc
	if(seg->GetLine()==false)
	{
#pragma warning("Logging");
//		G_LOG_1_ERROR("CyberShape %d is invalid! It's definition started with an arc segment!", _index);
		return false;
	}

	if(Panel::_debug)
	{
#pragma warning("Logging");
//		G_LOG_3_SOFTWARE("OddShapePart,#%d,Line(meters),%0.06lf,%0.06lf", _index, seg->GetPositionX(), seg->GetPositionY());
	}

	// Add the starting point to the list
	segmentVertices.push_back(Point(seg->GetPositionX(), seg->GetPositionY()));

	for(unsigned int i=1; i<_segments.size(); i++)
	{
		seg = _segments[i];
		Point prevPoint = segmentVertices.back();

		// Ignore segment if it is smaller than a pixel
		//if((fabs(seg->GetPositionX()-prevPoint.x)<pixelSize) &&
		//	(fabs(seg->GetPositionY()-prevPoint.y)<pixelSize))
		//	continue;

		// A counter clockwise arc is a concave feature
		if((seg->GetLine()==false) && (seg->GetClockwiseArc()==false))
			_concavePolygon = true;

		if(seg->GetLine()== true)
		{
			if(Panel::_debug)
			{
#pragma warning("Logging");
//				G_LOG_3_SOFTWARE("OddShapePart,#%d,Line(meters),%0.06lf,%0.06lf", _index, seg->GetPositionX(), seg->GetPositionY());
			}

			segmentVertices.push_back(Point(seg->GetPositionX(), seg->GetPositionY()));
		}
		else
		{
			// Use Jonathan Waltz's ArcPolygonizer to break up arcs to a series of points
			ArcPolygonizer arc(segmentVertices.back(), 
				               Point(seg->GetPositionX(), seg->GetPositionY()),
							   Point(seg->GetArcX(), seg->GetArcY()),
							   Point(0.0, 0.0),
							   0.0,
							   seg->GetClockwiseArc());

			// getPoints() returns all points on the arc, including the starting and ending points.
			PointList arcPoints = arc.getPoints();

			// start & end points are redundant.  remove them from the list
			arcPoints.pop_back();
			arcPoints.pop_front();

			for(PointList::iterator point=arcPoints.begin(); point!=arcPoints.end(); point++)
			{
				if(Panel::_debug)
				{
#pragma warning("Logging");
//					G_LOG_3_SOFTWARE("OddShapePart,#%d,Arc(meters),%0.06lf,%0.06lf", _index, point->x, point->y);
				}

				segmentVertices.push_back(Point(point->x, point->y));
			}
		}
	}



	//
	// Step 2: Check the points created in step 1 for duplicates.  Duplicates are 
	//         unnecessary and can cause the renderer to fail.
	//

	PointList::iterator vertex = segmentVertices.begin();
	for(; vertex!=segmentVertices.end(); vertex++)
	{
		bool duplicate = false;
		PointList::iterator point = _polygonPoints.begin();
		for(; point!=_polygonPoints.end(); point++)
		{
			// Check for duplicate points.  
			// This only checks for double precision changes.  Perhaps it should be larger?
			if( (fabs(vertex->x-point->x)<EPSILON) && (fabs(vertex->y-point->y)<EPSILON) )
			{
				duplicate = true;
				break;
			}
		}

		// If this is not a duplicate point, add to the classes list
		if(!duplicate)
		{
			if(Panel::_debug)
			{
#pragma warning("Logging");
//				G_LOG_3_SOFTWARE("OddShapePart,#%d,Vertex(meters),%0.06lf,%0.06lf", _index, vertex->x, vertex->y);
			}

			_polygonPoints.push_back((*vertex));
		}
	}


	//
	// Step 3: If the previous steps have not found this polygon to be 
	//         concave, walk through the angles and check that all turns
	//         are right turns.  If they are all right turns, it is convex.
	if(!_concavePolygon)
	{
		PointList::iterator A;
		PointList::iterator B;
		PointList::iterator C;

		// This is goofy looking, but works
		A = _polygonPoints.end();   // Not a point - special case of iterator
		//A--;						// Point[0] - First and Last points are the same
		C = _polygonPoints.begin();	// Point[0]

		A--;						// Point[-1]
		B = _polygonPoints.begin();	// Point[0]
		C++;						// Point[1]

		bool firstVertex = true;
		
		for(;C!=_polygonPoints.end(); B++, C++)
		{
			Point p0 = (*A);
			Point p1 = (*B)-p0;
			Point p2 = (*C)-p0;

			if(((p1.x*p2.y)-(p2.x*p1.y))>0) 
			{
				// CCW
				_concavePolygon = true;
				break;
			}


			if(firstVertex)
			{
				A = _polygonPoints.begin();
				firstVertex = false;
			}
			else
			{
				A++;
			}
		}
	}

	CalcAreaAndBound();
	InspectionArea();
	Feature::Validate(panelSizeX, panelSizeY);

	return true;
}

void CyberFeature::InspectionArea()
{
	if(_boundingBox==0)
		CalcAreaAndBound();

	InspectionAreaFromBounds();
}

void CyberFeature::CalcAreaAndBound()
{
	using namespace std;

	double x1,y1,x2,y2, cx,cy, C, A, R, theta;
	double area = 0;

	double minX = 9999999.9, maxX = -9999999.9, minY = 9999999.9, maxY = -9999999.9;
	double minCX, maxCX, minCY, maxCY;
	double XProd;
	bool inX = false;
	bool inY = false;

	// determine area for a polygon
	// no check for pen down
	for(unsigned int i=0; i<_segments.size()-1; i++)
	{
		CyberSegment *segment = _segments[i];
		x1 = segment->GetPositionX();
		y1 = segment->GetPositionY();

		// will rotate through on all in 1st position
		if(x1 > maxX)
			maxX = x1;
		if(x1 < minX)
			minX = x1;
		if(y1 > maxY)
			maxY = y1;
		if(y1 < minY)
			minY = y1;

		segment = _segments[i+1];
		x2 = segment->GetPositionX();
		y2 = segment->GetPositionY();
  
		// for polygon area calculation x1,y1 is current, x2,y2 is next in sequence  

		area += (x1*y2) - (y1*x2);
	}

	area = area/2.0;

	
	// add/subtract the arcs to the calculated area
	for(unsigned int i=0; i<_segments.size()-1; i++)
	{
		CyberSegment *segment = _segments[i];

		// if we're defining an arc
		if(!segment->GetLine() && i>0)
		{
			// for arc calculation x1,y1 is current point, x2,y2 is previous in sequence
			x1 = segment->GetPositionX();
			y1 = segment->GetPositionY();

			bool CW = segment->GetClockwiseArc();

			// get center point of possible arc
			cx = segment->GetArcX();
			cy = segment->GetArcY();

			// x1,y1 is the current, x2,y2 is the previous/earlier in sequence
			segment = _segments[i-1];
			x2 = segment->GetPositionX();
			y2 = segment->GetPositionY();

			C = sqrt( pow((x1-x2), 2) + pow((y1 - y2), 2) );
			R = sqrt( pow((x1-cx), 2) + pow((y1 - cy), 2) );

			if ( ((C / (2 * R)) <= 1) &&
				 ((C / (2 * R)) >= -1) )
			{
				// theta = 2 arcsin(C / 2*R)
				theta = 2 * asin( C / (2 * R) );

				// Area = r2[theta-sin(theta)]/2
				A = ( (pow(R, 2)) * ( theta - sin(theta) )) / 2; 

				// determine cross product -- placement of center up or down from chord
				 XProd = ((y2 - y1)*(cx - x1)) - ((cy - y1)*(x2 - x1));

				if(!CW)
				{
					if(XProd >= 0)
						area += A;
					else
						area += (PI * (pow(R, 2))) - A;
				}
				else
				{
					if(XProd >= 0)
						area -= (PI * (pow(R, 2))) - A;
					else
						area -= A;
				}


			// extents per arcs

				minCX = cx - R;
				maxCX = cx + R;

				minCY = cy - R;
				maxCY = cy + R;

				if (((x1 > cx) && (cx > x2)) || 
					((x2 > cx) && (cx > x1)))	
					inX = true;
				else
					inX = false;

				if (((y1 > cy) && (cy > y2)) || 
					((y2 > cy) && (cy > y1)))	
					inY = true;
				else
					inY = false;

				// Condition One
				if ( (inX == false) && (inY == false) )
				{
					if(XProd > 0)
					{
						if(CW == true)
						{
							maxX = max(maxCX, maxX);
							maxY = max(maxCY, maxY);
							minX = min(minCX, minX);
							minY = min(minCY, minY);
						}
						//else // CCW
					//	;	 // don't adjust	
					}
					else // XProd <= 0
					{
						if(CW == false)
						{
							maxX = max(maxCX, maxX);
							maxY = max(maxCY, maxY);
							minX = min(minCX, minX);
							minY = min(minCY, minY);
						}
					}
				} // end conditon one
				// condition Two	
				else if ((inX == true) && (inY == false))
				{
					if(XProd > 0)
					{
						if(CW == true)
						{
							if (x1 < x2)
							{
								minX = min(minCX, minX);
								maxX = max(maxCX, maxX);
								minY = min(minCY, minY);
							}
							else
							{
								minX = min(minCX, minX);
								maxX = max(maxCX, maxX);
								maxY = max(maxCY, maxY);
							}
						}
						else // CCW
						{	
							if (x1 < x2)
							{
								maxY = max(maxCY, maxY);
							}
							else
							{
								minY = min(minCY, minY);
							}
						}
					}
					else // XProd <= 0
					{
						if(CW == true)
						{
							if (x1 < x2)
							{
								minY = min(minCY, minY);
							}
							else
							{
								maxY = max(maxCY, maxY);
							}
						}
						else  //CCW
						{
							if (x1 < x2)
							{
								minX = min(minCX, minX);
								maxX = max(maxCX, maxX);
								maxY = max(maxCY, maxY);
							}
							else
							{
								minX = min(minCX, minX);
								maxX = max(maxCX, maxX);
								minY = min(minCY, minY);
							}
						}
					}
				}	// end condition Two

				// condition Three
				else if ((inX == false) && (inY == true))
				{
					if(XProd > 0)
					{
						if(CW == true)
						{
							if (y1 < y2)
							{
								maxX = max(maxCX, maxX);
								minY = min(minCY, minY);
								maxY = max(maxCY, maxY);
							}
							else
							{
								minX = min(minCX, minX);
								minY = min(minCY, minY);
								maxY = max(maxCY, maxY);
							}
						}
						else // CCW
						{
							if (y1 < y2)
							{
								minX = min(minCX, minX);
							}
							else
							{
								maxX = max(maxCX, maxX);
							}
						}
					}
					else // XProd <= 0
					{
						if(CW == true)
						{
							if (y1 < y2)
							{
								maxX = max(maxCX, maxX);
							}
							else
							{
								minX = min(minCX, minX);
							}
						}
						else // CCW
						{
							if (y1 < y2)
							{
								minX = min(minCX, minX);
								minY = min(minCY, minY);
								maxY = max(maxCY, maxY);
							}
							else
							{
								maxX = max(maxCX, maxX);
								minY = min(minCY, minY);
								maxY = max(maxCY, maxY);
							}
						}
					}
				}  // end condition Three
				
				// Condition Four
				else if ((inX == true) && (inY == true))
				{
					if(XProd > 0)
					{
						if (CW == true)
						{	
							if(y1 < y2)
							{
								maxX = max(maxCX, maxX);
							}
							else
							{
								minX = min(minCX, minX);
							}

							if (x1 < x2)
							{
								minY = min(minCY, minY);
							}
							else
							{
								maxY = max(maxCY, maxY);
							}
						}
						else // CCW
						{
							if (y1 < y2)
							{
								minX = min(minCX, minX);
							}
							else
							{
								maxX = max(maxCX, maxX);
							}
				
							if (x1 < x2)
							{
								maxY = max(maxCY, maxY);
							}
							else
							{
								minY = min(minCY, minY);
							}
						}
					}
					else // XProd <= 0
					{
						if(CW == true)
						{
							if (y1 < y2)
							{
								maxX = max(maxCX, maxX);
							}
							else
							{
								minX = min(minCX, minX);
							}

							if (x1 < x2)
							{
								minY = min(minCY, minY);
							}
							else
							{
								maxY = max(maxCY, maxY);
							}
						}
						else // CCW
						{
							if (y1 < y2)
							{
								minX = min(minCX, minX);
							}
							else
							{
								maxX = max(maxCX, maxX);
							}

							if (x1 < x2)
							{
								maxY = max(maxCY, maxY);
							}
							else
							{
								minY = min(minCY, minY);
							}
						}
					}
				} // end condition Four
			}
		}
		
	}

	// take the absolute value
	area = fabs(area);
	_nominalArea = area;

	_boundingBox.p1.x = minX;
	_boundingBox.p1.y = minY;
	_boundingBox.p2.x = maxX;
	_boundingBox.p2.y = maxY;
}