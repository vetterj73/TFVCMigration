/*
	A feature is a single pad

	Is a runtime object.  Therefore, contains state information
*/

#pragma once

#include "Box.h"
#include <string>
#include <vector>
using std::string;
using std::vector;

class Panel;

class Feature
{
public:	

	typedef enum {
		FEATURE_UNINITIALIZED =	0,
		FEATURE_UNINSPECTED,
		FEATURE_INSPECTED
	} FeatureStatus ;

	typedef enum {
		SHAPE_UNDEFINED = -1,
		SHAPE_CHECKERPATTERN,
		SHAPE_CROSS,
		SHAPE_DIAMOND,
		SHAPE_DIAMONDFRAME,
		SHAPE_DISC,
		SHAPE_DONUT,
		SHAPE_RECTANGLE,
		SHAPE_RECTANGLEFRAME,
		SHAPE_TRIANGLE,
		SHAPE_EQUILATERALTRIANGLEFRAME,
		SHAPE_CYBER
	} PadShape;

	Feature();
	virtual ~Feature();

	FeatureStatus GetStatus()		{ return _status; }
	void SetStatus(FeatureStatus s) { _status=s; }

	PadShape GetShape()		{ return _shape; }

	// description, in CAD space
	double GetCadX()		{ return _xCadCenter; }
	double GetCadY()		{ return _yCadCenter; }
	double GetRotation()	{ return _rotation;   } // degrees

	// inspection information
	Box GetBoundingBox()	{ return _boundingBox;    }
	Box GetInspectionArea()	{ return _inspectionArea; }
	double GetNominalArea()	{ return _nominalArea;    }

	unsigned short GetApertureValue()			{ return _apertureValue;  }
	void SetApertureValue(unsigned short a)	{ _apertureValue = a; }

	string GetName()		{ return _name;  }
	unsigned int GetId()	{ return _index; }

	double GetPercentagePadCoverage(){ return _percentagePadCoverage;}
	double GetXPasteCenter(){ return _xPasteCenter;}
	double GetYPasteCenter(){ return _yPasteCenter;}
	double GetBridgeWidth(){ return _bridgeWidth;}

	void SetResults(double coveragePercent, double bridgeWidth, double xCenter, double yCenter);
	void ResetResults();

	virtual bool Validate(Panel *pPanel);

protected:
	Feature(PadShape shape, int id, double positionX, double positionY, double rotation);

	void InspectionAreaFromBounds();

	//
	// Feature info to be filled by derived shape classes
	PadShape		_shape;

	string			_name;
	unsigned int	_index;
	
	// Results
	double			_percentagePadCoverage;
	double			_xPasteCenter;
	double			_yPasteCenter;
	double			_bridgeWidth;

	// description, in CAD space
	double			_xCadCenter; 
	double			_yCadCenter;
	double			_rotation; // degrees

	// Inspection information
	FeatureStatus	_status;
	Box				_boundingBox;
	Box				_inspectionArea;
	double			_nominalArea;
	unsigned short	_apertureValue;
};

class CrossFeature : public Feature
{
public:
	CrossFeature(int id, double positionX, double positionY, double rotation,
				 double sizeX, double sizeY, double legSizeX, double legSizeY );
	~CrossFeature();

	double GetSizeX()		{ return _sizeX; }
	double GetSizeY()		{ return _sizeY; }
	double GetLegSizeX()	{ return _legSizeX; }
	double GetLegSizeY()	{ return _legSizeY; }

	const PointList& GetPointList( ) const { return _polygonPoints; }

private:
	// Methods
	void Bound();
	void NominalArea();
	void InspectionArea();

	// Shape Definition Parameters
	double _sizeX;
	double _sizeY;
	double _legSizeX;
	double _legSizeY;

	// Shape vertices
	PointList _polygonPoints;
};

class DiamondFeature : public Feature
{
public:

	DiamondFeature(int id, double positionX, double positionY, double rotation,
				   double sizeX, double sizeY );
	~DiamondFeature();

	double GetSizeX()	{ return _sizeX; }
	double GetSizeY()	{ return _sizeY; }

	const PointList& GetPointList( ) const { return _polygonPoints; }

protected:
	// Shape Definition Parameters
	double _sizeX;
	double _sizeY;

private:
	// Methods
	void Bound();
	virtual void NominalArea();
	void InspectionArea();

	// Shape vertices
	PointList _polygonPoints;
};

class DiamondFrameFeature : public DiamondFeature
{
public:

	DiamondFrameFeature(int id, double positionX, double positionY, double rotation,
				     double sizeX, double sizeY, double thickness);
	~DiamondFrameFeature();

	double GetThickness()   { return _thickness; };

	const PointList& GetInnerPointList( ) const { return _innerPolygonPoints; }

private:
	// Methods
	void NominalArea(); // Override function
	void CalInnerPolygon();

	double _thickness;

	// Shape vertices
	PointList _innerPolygonPoints;
};

class DiscFeature : public Feature
{
public:

	DiscFeature(int id, double positionX, double positionY,	double diameter);
	~DiscFeature();

	double GetDiameter()	{ return _diameter; }

private:
	// Methods
	void Bound();
	void NominalArea();
	void InspectionArea();

	// Shape Definition Parameters
	double _diameter;
};

class DonutFeature : public Feature
{
public:

	DonutFeature(int id, double positionX, double positionY, double diameterInside, double diameterOutside);
	~DonutFeature();

	double GetDiameterInside()	{ return _diameterInside; }
	double GetDiameterOutside()	{ return _diameterOutside; }

private:
	// Methods
	void Bound();
	void NominalArea();
	void InspectionArea();

	// Shape Definition Parameters
	double _diameterInside;
	double _diameterOutside;
};
	
class RectangularFeature : public Feature
{
public:

	RectangularFeature(int id, double positionX, double positionY, double rotation,
				     double sizeX, double sizeY );
	~RectangularFeature();

	double GetSizeX()	{ return _width; }
	double GetSizeY()	{ return _height; }

	const PointList& GetPointList( ) const { return _polygonPoints; }

protected:
	// Shape Definition Parameters
	double _width;
	double _height;

private:
	// Methods
	void Bound();
	virtual void NominalArea();
	void InspectionArea();

	// Shape vertices
	PointList _polygonPoints;
};

class RectangularFrameFeature : public RectangularFeature
{
public:

	RectangularFrameFeature(int id, double positionX, double positionY, double rotation,
				     double sizeX, double sizeY, double thickness );
	~RectangularFrameFeature();

	double GetThickness()   { return _thickness; };

	const PointList& GetInnerPointList( ) const { return _innerPolygonPoints; }

private:
	// Methods
	void NominalArea(); // Override function
	void CalInnerPolygon();

	double _thickness;

	// Shape vertices
	PointList _innerPolygonPoints;
};

class TriangleFeature : public Feature
{
public:

	TriangleFeature(int id, double positionX, double positionY, double rotation,
					double sizeX, double sizeY, double offsetX );
	~TriangleFeature();

	double GetSizeX()	{ return _sizeX; }
	double GetSizeY()	{ return _sizeY; }
	double GetOffset()	{ return _offset; }

	const PointList& GetPointList( ) const { return _polygonPoints; }

protected:
	// Shape Definition Parameters
	double _sizeX;
	double _sizeY;
	double _offset;

private:
	// Methods
	void Bound();
	virtual void NominalArea();
	void InspectionArea();

	// Shape vertices
	PointList _polygonPoints;
};

class EquilateralTriangleFrameFeature : public TriangleFeature
{
public:

	EquilateralTriangleFrameFeature(int id, double positionX, double positionY, double rotation,
				     double size, double thickness );
	~EquilateralTriangleFrameFeature();

	double GetThickness()   { return _thickness; };

	const PointList& GetInnerPointList( ) const { return _innerPolygonPoints; }

private:
	// Methods
	void NominalArea(); // Override function
	void CalInnerPolygon();

	double _thickness;

	// Shape vertices
	PointList _innerPolygonPoints;
};

class CheckerPatternFeature : public Feature
{
public:

	CheckerPatternFeature(int id, double positionX, double positionY, double rotation,
					double sizeX, double sizeY);
	~CheckerPatternFeature();

	double GetSizeX()	{ return _sizeX; }
	double GetSizeY()   { return _sizeY; }

	const PointList& GetFirstPointList() const { return _polygonPoints[0]; }
	const PointList& GetSecondPointList() const { return _polygonPoints[1]; }

protected:
	// Shape Definition Parameters
	double _sizeX;
	double _sizeY;

private:
	// Methods
	void Bound();
	virtual void NominalArea();
	void InspectionArea();

	// Shape vertices
	PointList _polygonPoints[2];
};

class CyberSegment
{
public:

	CyberSegment(bool line, bool penDown, bool clockwiseArc, double positionX,
				 double positionY, double arcX, double arcY);
	~CyberSegment() {}

	bool GetLine()			{ return _line; }
	bool GetPenDown()		{ return _penDown; }
	bool GetClockwiseArc()	{ return _clockwiseArc; }
	double GetPositionX()	{ return _positionX; }
	double GetPositionY()	{ return _positionY; }
	double GetArcX()		{ return _arcX; }
	double GetArcY()		{ return _arcY; }

private:
	bool _line;
	bool _penDown;
	bool _clockwiseArc;
	double _positionX;
	double _positionY;
	double _arcX;
	double _arcY;
};

class CyberFeature : public Feature
{
public:

	CyberFeature(int id, double positionX, double positionY, double rotation);
	~CyberFeature();

	void AddSegment(CyberSegment *segment);

	CyberSegment* GetFirstSegment();
	CyberSegment* GetNextSegment();

	int GetNumberSegments();

	// SRF files containing CyberShapes have had problems.
	// Validate will check for duplicate points and type
	// of polygon.  These features are very special cases.
	bool      IsConcave(); 
	bool      Validate(Panel *pPanel);

	const PointList& GetPointList( ) const { return _polygonPoints; }

private:
	// Methods
	void CalcAreaAndBound();
	//void NominalArea();
	void InspectionArea();

	// Shape Definition Parameters
	vector<CyberSegment*> _segments;

	vector<CyberSegment*>::iterator _currentSegment;

	bool _concavePolygon;

	// Shape vertices
	PointList _polygonPoints;
};