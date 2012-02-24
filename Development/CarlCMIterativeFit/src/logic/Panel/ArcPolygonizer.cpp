#include "ArcPolygonizer.h"

#include <cmath>
#include <sstream>

#include <stdio.h>

const double ArcPolygonizer::DEFAULT_TOLERANCE = 0.000015*0.3;
const double ArcPolygonizer::EPSILON = 0.0000001;
const double ArcPolygonizer::ARC_EPSILON = 0.000015*0.3;

ArcPolygonizer::ArcPolygonizer(
	Point  start,
	Point  end,
	Point  center,
	Point  offset,
	double theta,
	bool   clockwise,
	double tolerance )
{
	set( start, end, center, offset, theta, clockwise );
	setTolerance( tolerance );
}

void ArcPolygonizer::set(
	Point  start,
	Point  end,
	Point  center,
	Point  offset,
	double theta,
	bool   clockwise )
{
	m_start     = start;
	m_end       = end;
	m_center    = center;
	m_offset    = offset;
	m_theta     = theta;
	m_clockwise = clockwise;
	m_valid     = false;
}

void ArcPolygonizer::setTolerance(
	double tolerance )
{
	m_tolerance = tolerance;
	m_valid     = false;
}

const std::list<Point>& ArcPolygonizer::getPoints()
{
	if( !m_valid )
	{
		calc();
		m_valid = true;
	}
	return m_points;
}

double ArcPolygonizer::ValidatePoints(
	PointList points,
	Point     center )
{
	bool first = true;
	double minr2 = 0., maxr2=0; // radius squared min and max
	for( PointList::const_iterator p=points.begin(); p!=points.end(); ++p )
	{
		Point p2c = *p - center;
		p2c *= p2c;
		double r2 = p2c.x + p2c.y;
		if( first )
		{
			minr2 = maxr2 = r2;
			first = false;
		}
		else
		{
			if( minr2 > r2 ) minr2 = r2;
			else if( maxr2 < r2 ) maxr2 = r2;
		}
	}
	return maxr2-minr2;
}

double ArcPolygonizer::XY2Theta(
	Point  point,
	double radius )
{
	if(radius == 0) return 99999.9;

	double theta;

	//
	// asin is only valid from -1.0 to 1.0, check for rounding errors
	//
	double temp = std::abs(point.y)-std::abs(radius);
	if(temp < 0.0)
	{
		theta = std::asin(point.y/radius);
		if(point.y >= 0)
		{
			if(point.x < 0)
			{
				theta = PI - theta;
			}
		}
		else
		{
			if(point.x < 0)
			{
				theta = PI + std::abs(theta);
			}
			else
			{
				theta = 2 * PI - std::abs(theta);
			}
		}
	}
	else
	{
		// when they are the same, just return one half PI,
		// determine sign by testing y, r should always be positive.
		if(point.y > 0)
			theta = PI / 2.0;
		else
			theta = 3.0 * PI / 2.0;
	}

	return theta;
}

void ArcPolygonizer::CalcIntervals(
	Point   start,
	Point   end,
	double  radius,
	double  tolerance,
	bool    clockwise,
	double& interval,
	double& nSegments )
{
	bool segmented = false;

	double theta, theta1, theta2;
	theta1 = std::atan2( start.y, start.x ); // XY2Theta(start, radius);
	theta2 = std::atan2( end.y, end.x );     // XY2Theta(end, radius);
	if(theta2 > theta1)
		theta = std::fmod(PI * 2.0 + theta1 - theta2,PI * 2.0) ;
	else
		theta = theta1 - theta2;

	do // only one iteration - broken out if we decide NOT to segment the arc
	{
		if(radius == 0) break;
		if(tolerance >= radius ) break;

		//theta = theta2 - theta1;
		theta = clockwise ? -std::abs(theta) : std::abs(2 * PI - theta);

		interval = 2 * std::acos(1 - tolerance/radius);
		if(interval == 0) break;

		nSegments = int(std::abs(theta/interval) + 0.5);
		if(nSegments == 0) break;

		interval = theta/nSegments;
		segmented = true;
	}
	while(false);

	if( !segmented )
	{
		nSegments = 1;
		interval = theta;
	}
//	printf(" start %lf,%lf(%lf) end %lf,%lf(%lf) th %lf : %lfx%lf\n",
//		start.x, start.y, theta1 / (2.*PI),
//		end.x, end.y, theta2 / (2.*PI),
//		theta  / (2.*PI), interval / (2.*PI), nSegments );
}

void ArcPolygonizer::CalcPoints(
	Point      start,
	Point      end,
	double     radius,
	double     interval,
	int        nSegments,
	PointList& nodes )
{
	if( radius != 0 || nSegments != 1 )
	{
		double theta1 = std::atan2( start.y, start.x ); // XY2Theta(start, radius);

		nodes.push_back( start );
		Point firstPoint(
			radius * std::cos(theta1 + interval),
			radius * std::sin(theta1 + interval) );
		nodes.push_back( firstPoint );
		Point delta = firstPoint - start;
		for(int i = 2; i <= nSegments; i++)
		{
			delta.rot( interval );
			nodes.push_back( nodes.back() + delta );
		}
	}
	nodes.push_back( end );
}

void ArcPolygonizer::calc()
{
	// apply rotation to points
	// note that m_start has already been rotated as it is
	// a part of the previous segment/arc/startpoint
	double theta1 = m_theta * 2 * PI; //theta is % of -2PI to 2PI

	m_end -= m_offset;
	m_end.rot( theta1 );
	m_end += m_offset;

	m_center -= m_offset;
	m_center.rot( theta1 );
	m_center += m_offset;

	double deviation = ValidatePoints( m_start, m_end, m_center);
	if( deviation > EPSILON )
	{
		std::ostringstream es;
		es << "Cybershape arc start and end points are on arcs ";
		es << std::sqrt(deviation) << ">" << std::sqrt(EPSILON) << " apart. ";
		es << "offset (" << m_offset.x << "," << m_offset.y << ") ";
		es << "center (" << m_center.x << "," << m_center.y << ") ";
		es << "start (" << m_start.x << "," << m_start.y << ") ";
		es << "end (" << m_end.x << "," << m_end.y << ") ";
		throw std::runtime_error( es.str().c_str() );
	}

	m_start -= m_center;
	m_end   -= m_center;
	m_radius2 = m_start.x*m_start.x + m_start.y*m_start.y;
	m_radius = std::sqrt( m_radius2 );
	CalcIntervals(
		m_start, m_end, m_radius,
		m_tolerance, m_clockwise,
		m_interval, m_nSegments );
	CalcPoints(
		m_start, m_end, m_radius,
		m_interval, (int)m_nSegments,
		m_points );
	for( PointList::iterator i=m_points.begin(); i!=m_points.end(); ++i )
		*i += m_center;
}
