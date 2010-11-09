#pragma once 

#include "Point.h"
#include "STL.h"

#include <math.h>

const double PI = 3.14159265359;

class ArcPolygonizer
{
public:
	static const double DEFAULT_TOLERANCE;
	static const double EPSILON;
	static const double ARC_EPSILON;

	ArcPolygonizer(
		Point  start     = Point(),
		Point  end       = Point(),
		Point  center    = Point(),
		Point  offset    = Point(),
		double theta     = 0.,
		bool   clockwise = true,
		double tolerance = DEFAULT_TOLERANCE );

	void set(
		Point  start,
		Point  end,
		Point  center,
		Point  offset,
		double theta,
		bool   clockwise );

	void setTolerance(
		double tolerance = DEFAULT_TOLERANCE );

	const list<Point>& getPoints();

protected:
	///
	/// validate whether supplied points are equadistant from the center point
	/// and are therefore part of the same circular arc.
	///
	static double ValidatePoints(
		PointList points,
	 	Point     center );

	inline static double ValidatePoints( Point p1, Point p2, Point center )
	{ PointList p; p.push_back(p1); p.push_back(p2); return ValidatePoints(p,center); }

	///
	/// extract theta information of the polar system from given X, Y, R(center (0, 0))
	///
	static double XY2Theta(
		Point  point,
		double radius );

	///
	/// given the tolerance, calculate the number of segments and angle interval of arc
	///
	static void CalcIntervals(
		Point   start,     ///< @param[in]  arc start point
		Point   end,       ///< @param[in]  arc end point
		double  radius,    ///< @param[in]  arc radius
		double  tolerance, ///< @param[in]  max length of arc segment to use
		bool    clockwise, ///< @param[in]  clockwise/counterclockwise
		double& interval,  ///< @param[out] angular interval (in radians)
		double& nSegments  ///< @param[out] number of segments in arc
		);

	///
	/// Given the inputs, create list of points on the arc
	///
	static void CalcPoints(
		Point      start,
		Point      end,
		double     radius,
		double     interval,
		int        nSegments,
		PointList& nodes       ///< @param[out] list of nodes
		);

private:
	///
	/// convert arc to polygon segments, the outputs are polygon nodes
	///
	/// @throws std::runtime_error
	void calc();

	/// input variables
	Point     m_start;     ///< arc startpoint
	Point     m_end;       ///< arc endpoint
	Point     m_center;    ///< arc centerpoint
	Point     m_offset;    ///< arc offset
	double    m_theta;     ///< arc rotation
	bool      m_clockwise; ///< arc direction

	/// intermediate variables
	double    m_radius2;
	double    m_radius;
	double    m_tolerance; ///< angular tolerance between arc segments
	double    m_interval;  ///< angle interval for each segment of arc
	double    m_nSegments; ///< number of segment

	/// output variables
	bool      m_valid;     ///< pointlist is valid?
	PointList m_points;    ///< pointlist output
};
