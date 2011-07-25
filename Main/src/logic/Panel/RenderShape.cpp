#include "ArcPolygonizer.h"
#include "RenderShape.h"
#include "Utilities.h"

// Rudd includes
#include "aapoly.h"
#include "nint.h"

#include "..\..\..\RuddLib\include\feature.h"
using RUDD::FEATURE_ARGS;
using RUDD::FEAT_CIRC;
using RUDD::FEAT_RECT;
using RUDD::SCONV_ARGS;


//
//
// The following templated function will render discs and rects in
// either 8 or 16 bit images and with or without anti-aliasing.
//
// Complex polygon rendering is implemented below.
template <typename IMAGETYPE>
void RenderShape(IMAGETYPE& image, double resolution, 
				 unsigned int featGrayValue, int featType, int antiAlias,
				 double height, double width, double rotation, double diameter, 
				 double xCenter, double yCenter)
{
	int nrows(-1), ncols(-1), i(-1);

	double radius = diameter / 2.0;

	if(antiAlias)
		featGrayValue = (1<<(image.GetBitsPerPixel()))-1;  // Set gray to 255 for 8 bit image, 65535 for 16 bit images

	RUDD::SCONV_ARGS sconvArgs;
	RUDD::SCInit(&sconvArgs);

	sconvArgs.AAtype = antiAlias;
	sconvArgs.featz = (double) featGrayValue;
	sconvArgs.featt = rotation / 360.;
	if (featType == FEAT_CIRC) 
	{
		sconvArgs.FeatType = FEAT_CIRC;
		sconvArgs.featw = radius/resolution;
		sconvArgs.feath = 0.;
	} 
	else if (featType == FEAT_RECT)
	{
		sconvArgs.FeatType = FEAT_RECT;
		radius = sqrt(width*width + height*height)/2; //distance to corners
		sconvArgs.feath = height/resolution/2.; // half height and width
		sconvArgs.featw = width/resolution/2.;
		sconvArgs.featt = rotation / 360.; //fraction of a circle
	} 
   
	// calculate the bounding box of the feat, in Cad space
	double xMin	= xCenter - radius;
	double xMax	= xCenter + radius;
	double yMin	= yCenter - radius; // oriented in CAD space
	double yMax	= yCenter + radius; // 
	

	double centerCol, centerRow, dTempRow, dTempCol;
	unsigned int firstCol, firstRow, lastCol, lastRow;
	image.WorldToImage(xCenter, yCenter, &centerRow, &centerCol);
	image.WorldToImage(xMin, yMin, &dTempRow, &dTempCol);
	firstRow = (unsigned int)dTempRow;
	firstCol = (unsigned int)dTempCol;
	image.WorldToImage(xMax, yMax, &dTempRow, &dTempCol);
	lastRow = (unsigned int)dTempRow;
	lastCol = (unsigned int)dTempCol;

	if(antiAlias)
	{
		// increase size to allow for antialiasing
		firstCol-=2;
		firstRow-=2;  
		lastCol+=3;  
		lastRow+=3;
	}

	nrows = lastRow - firstRow + 1;  //lastRow is INCLUSIVE!!!!
	ncols = lastCol - firstCol + 1;

	if (firstCol <0 || firstRow < 0 || 
		lastCol >= image.Columns() || lastRow >= image.Rows())
	{
		//G_LOG_0_ERROR("trying to Render pad outside of CAD image");
		return;
	}


	// Specify the fraction of pixel the centriod is away from the center of the
	// area in the array provided
	sconvArgs.featx = centerCol - (firstCol + lastCol)/2.0;
	sconvArgs.featy = centerRow - (firstRow + lastRow)/2.0;
	
	if(image.GetBytesPerPixel() == 2)
	{
		unsigned short* buf = (unsigned short*)image.GetBuffer();
		buf += firstRow*image.PixelRowStride()+firstCol;
		ScanConv(ncols, nrows, buf, 
			image.PixelRowStride(), sconvArgs);
	}
	else
		ScanConv(ncols, nrows, image.GetBuffer(firstCol, firstRow), 
			image.PixelRowStride(), sconvArgs);
}

//
//
// Discs
//
//
template <typename IMAGETYPE>
void RenderDisc(IMAGETYPE& image, double resolution, DiscFeature* disc, unsigned int grayValue, int antiAlias)
{
	RenderShape(image, resolution, grayValue, FEAT_CIRC, antiAlias, 0, 0, 0, disc->GetDiameter(), disc->GetCadX(), disc->GetCadY());
}

// Create Image and Image16 instances
template void RenderDisc(Image& image, double resolution, DiscFeature* disc, unsigned int grayValue, int antiAlias);
template void RenderDisc(Image16& image, double resolution, DiscFeature* disc, unsigned int grayValue, int antiAlias);



//
//
// Donuts
//
//
template <typename IMAGETYPE>
void RenderDonut(IMAGETYPE& image, double resolution, DonutFeature* donut, unsigned int grayValue, int antiAlias)
{
	// Get the coordinates to where the donut is in the image
	Box box = donut->GetBoundingBox();

	RenderShape(image, resolution, grayValue, FEAT_CIRC, antiAlias, 0, 0, 0, donut->GetDiameterOutside(), donut->GetCadX(), donut->GetCadY());

	// Have to draw the donut hole in a separate image and substract it from the disc drawn on the mask
	double width = box.Width();
	double height = box.Height();
	int cols = (int) (width/resolution+0.5);
	int rows = (int) (height/resolution+0.5);

	ImgTransform trans;
	trans.Config(resolution, resolution);

	// Set up image to draw donut hole
	IMAGETYPE donutHole;
	donutHole.Configure(
		cols, 
		rows, 
		cols,
		trans,
		trans,
		true);	// create own buffer

	// Create donut hole
	RenderShape(donutHole, resolution, grayValue, FEAT_CIRC, antiAlias, 0, 0, 0, donut->GetDiameterInside(), width/2.0, height/2.0);

	//string filename = Config::instance().getImageDir();
	//filename += "\\DonutHole.bmp";
	//Bitmap* bmp = Bitmap::NewBitmapFromBuffer(	donutHole.Rows(),
	//											donutHole.Columns(),
	//											donutHole.RowStride(),
	//											donutHole._buffer,
	//											donutHole._bitsPerPixel	);
	//if(bmp) bmp->write(filename);
	//delete bmp;

	// Transform to pixel coordinates, and position them on nearest whole integer pixels
	Box temp;
	donutHole.WorldToImage(box.p1.x, box.p1.y, &temp.p1.y, &temp.p1.x);
	donutHole.WorldToImage(box.p2.x, box.p2.y, &temp.p2.y, &temp.p2.x);
	box = temp;
	box.p1.x = NearestIntD(box.p1.x);
	box.p1.y = NearestIntD(box.p1.y);
	box.p2.x = NearestIntD(box.p2.x);
	box.p2.y = NearestIntD(box.p2.y);

	// Remove donut hole from donut.
	int dPix, iPix;
	for(int r=0, ri=(int)box.p1.y; r<rows; r++, ri++)
	{
		for(int c=0, ci=(int)box.p1.x; c<cols; c++, ci++)
		{
			dPix = r*donutHole.Columns()+c;
			iPix = ri*image.Columns()+ci;

			if(image.GetBytesPerPixel() == 1)
				image.GetBuffer()[iPix]-=donutHole.GetBuffer()[dPix];
			else
				((unsigned short*)image.GetBuffer())[iPix]-=((unsigned short*)donutHole.GetBuffer())[dPix];
		}
	}
}

// Create Image and Image16 instances
template void RenderDonut(Image& image, double resolution, DonutFeature* donut, unsigned int grayValue, int antiAlias);
template void RenderDonut(Image16& image, double resolution, DonutFeature* donut, unsigned int grayValue, int antiAlias);



//
//  Functions for drawing polygons
//



//
//  Anti-aliased polygon renderer
//
template<typename IMAGETYPE>
void RenderAAPolygon(IMAGETYPE& image, Feature* feature, PointList& polygonPoints)
{
	// polygonPoints should be in CAD space and in a CW order.  
	// The tranformation of the CAD points to pixel points will transpose them to a CCW.
	// If the points provided are in CCW order, they will be transposed to a CW shape, which will be treated as a hole
	// Convert to pixel coordinates and Rudd's structure.
	double pixelX, pixelY;

	int numPoints = (int) polygonPoints.size();
	POLYVERT *p = new POLYVERT[numPoints];

	for(int i=0; i<numPoints; i++)
	{
		Point pt;

		// get first point
		pt = polygonPoints.front();

		// Transform point to pixels
		image.WorldToImage(pt.x, pt.y, &pixelY, &pixelX);

		p[i].x = pixelX;
		p[i].y = pixelY;
		p[i].dx = 0.0;
		p[i].dy = 0.0;

		// remove the point from the list
		polygonPoints.pop_front();

		//G_LOG_3_SYSTEM("OddShapePart,#%d,Point(Pixels),%0.06lf,%0.06lf", feature->_index, p[i].x, p[i].y);
	}

	//
	// Pass the points to aapoly, they must be CCW to draw the shape.  CW will create holes.
	// The "transformation" of the point (CADToImage) is a reflection from X to Y, not a rotation.
	// Hence, the points that are CW from the CPad object, are converted to CCW due to the transformation.
	int returnValue = 0;
	
	if(image.GetBytesPerPixel()==1)
		aapoly(image.Columns(), image.Rows(), image.GetBuffer(), image.Columns(), numPoints, p);
	else
		aapoly(image.Columns(), image.Rows(), (unsigned short*)image.GetBuffer(), image.Columns(), numPoints, p);

	delete[] p;

	if(returnValue != 0)
	{
		//G_LOG_1_ERROR("aapoly returned %d", returnValue);
	}
}

// Create Image and Image16 instances
template void RenderAAPolygon(Image& image, Feature* feature, PointList& polygonPoints);
template void RenderAAPolygon(Image16& image, Feature* feature, PointList& polygonPoints);



//
//
// Polygon rendering.  Redirects to RenderAAPolygon when the antiAlias flag is set.  
// Otherwise, it will pass a separate image to RenderAAPolygon, then copy image 
// into destination image while converting it into a two level feature of specified grayValue.
//
//
template<typename IMAGETYPE>
void RenderPolygon(IMAGETYPE& image, double resolution, 
				   Feature* feature, PointList polygonPoints[], int numPolygons, 
				   unsigned int grayValue, int antiAlias)
{
	if(antiAlias)
	{
		for(int i=0; i<numPolygons; i++)
			RenderAAPolygon(image, feature, polygonPoints[i]);
		return;
	}

	// Get the coordinates to where the polygon is in the image
	Box roi = feature->GetInspectionArea();

	// Transform to pixel coordinates, and position them on nearest whole integer pixels
	Box temp;
	image.WorldToImage(roi.p1.x, roi.p1.y, &temp.p1.y, &temp.p1.x);
	image.WorldToImage(roi.p2.x, roi.p2.y, &temp.p2.y, &temp.p2.x);
	roi = temp;
	roi.p1.x = NearestIntD(roi.p1.x);
	roi.p1.y = NearestIntD(roi.p1.y);
	roi.p2.x = NearestIntD(roi.p2.x);
	roi.p2.y = NearestIntD(roi.p2.y);

	// @todo The check of inspection area should occur when feature is created.
	// Check that the ROI is within the image
	if(((int) roi.p1.x) < 0)
		roi.p1.x = 0.0;
	
	if(((int) roi.p1.y) < 0)
		roi.p1.y = 0.0;

	if(((int) roi.p2.x) > ((int)image.Columns()-1))
		roi.p2.x = image.Columns()-1;

	if(((int) roi.p2.y) > ((int)image.Rows()-1))
		roi.p2.y = image.Rows()-1;


	// Have to draw the feature in a separate image and add it to the mask
	int cols = (int) roi.Width();
	int rows = (int) roi.Height();

	// Set up image to draw polygon
	Image16 polygonImage;
	polygonImage.Configure(cols, rows, cols, true);


	for(int i=0; i<numPolygons; i++)
	{
		// Convert the points to fit inside the new image
		PointList shapePoints;
		for(PointList::iterator p=polygonPoints[i].begin(); p!=polygonPoints[i].end(); p++)
		{
			Point point;
			point.x = (*p).x-feature->GetInspectionArea().p1.x;
			point.y = (*p).y-feature->GetInspectionArea().p1.y;
			shapePoints.push_back(point);
		}


		// Create shape
		RenderAAPolygon(polygonImage, feature, shapePoints);
	}

	//char filename[100];
	//_snprintf(filename, 100, "%s\\Polygon-%d.tif", Config::instance().getImageDir().c_str(), feature->_index);
	//saveMonoTifFile(filename, polygonImage._buffer, polygonImage.Columns(), polygonImage.Rows(), polygonImage._bitsPerPixel);


	// Add shape to destination image.  This will leave other features that
	// might lie within the bounding box along.  However, overlapping features
	// will have problems.
	int iPix, pPix;
	int threshold = (int) ((1<<(polygonImage.GetBitsPerPixel()))/2.0);
	unsigned int gv;
	for(int r=0, ri=(int)roi.p1.y; r<rows; r++, ri++)
	{
		for(int c=0, ci=(int)roi.p1.x; c<cols; c++, ci++)
		{
			pPix = r*polygonImage.Columns()+c;
			iPix = ri*image.Columns()+ci;
			gv = (polygonImage.GetBuffer16()[pPix]>threshold)*grayValue;
			unsigned short* buffer = (unsigned short*)image.GetBuffer();
			buffer[iPix] += gv;
		}
	}
}

// Create Image and Image16 instances
template void RenderPolygon(Image& image, double resolution, Feature* feature, PointList polygonPoints[], int numPolygons, unsigned int grayValue, int antiAlias);
template void RenderPolygon(Image16& image, double resolution, Feature* feature, PointList polygonPoints[], int numPolygons, unsigned int grayValue, int antiAlias);



//
//
// CyberShapes
//
//
template <typename IMAGETYPE>
void RenderCyberShape(IMAGETYPE& image, double resolution, CyberFeature* cyberShape, unsigned int grayValue, int antiAlias)
{
#if 0 
	bool concavePolygon = false;
	// polygonPoints needs to be in meters for ArcPolygonizer
	PointList polygonPoints;

	vector<CyberSegment>::iterator seg = cyberShape->_segments.begin();

	// The first segment must not be an arc
	if(seg->_line==false)
	{
		G_LOG_1_ERROR("CyberShape %d is invalid! It's definition started with an arc segment!", cyberShape->_index);
		return;
	}

	// Add the starting point to the list
	polygonPoints.push_back(Point(seg->_positionX, seg->_positionY));

	// Start the for loop on the second segment
	seg++;  

	for(; seg!=cyberShape->_segments.end(); seg++)
	{
		Point prevPoint = polygonPoints.back();

		// Ignore segment if it is smaller than a pixel
		if((fabs(seg->_positionX-prevPoint.x)<resolution) &&
			(fabs(seg->_positionY-prevPoint.y)<resolution))
			continue;

		// A counter clockwise arc is a concave feature
		if((seg->_line==false) && (seg->_clockwiseArc==false))
			concavePolygon = true;

		if(seg->_line == true)
		{
			//G_LOG_3_SYSTEM("OddShapePart,#%d,Line(meters),%0.06lf,%0.06lf", cyberShape->_index, seg->_positionX, seg->_positionY);
			polygonPoints.push_back(Point(seg->_positionX, seg->_positionY));
		}
		else
		{
			ArcPolygonizer arc(polygonPoints.back(), 
				               Point(seg->_positionX, seg->_positionY),
							   Point(seg->_arcX, seg->_arcY),
							   Point(0.0, 0.0),
							   0.0,
							   seg->_clockwiseArc);

			// getPoints() returns all points on the arc, including the starting and ending points.
			PointList arcPoints = arc.getPoints();

			// start & end points are redundant.  remove them from the list
			arcPoints.pop_back();
			arcPoints.pop_front();

			for(PointList::iterator point=arcPoints.begin(); point!=arcPoints.end(); point++)
			{
				//G_LOG_3_SYSTEM("OddShapePart,#%d,Arc(meters),%0.06lf,%0.06lf", cyberShape->_index, point->x, point->y);
				polygonPoints.push_back(Point(point->x, point->y));
			}
		}
	}

	// If the polygon did not contain a CCW arc, check that all turns are right turns
	if(!concavePolygon)
	{
		PointList::iterator A;
		PointList::iterator B;
		PointList::iterator C;

		// This is very goofy
		A = polygonPoints.end();
		A--;						// Point[0] - First and Last points are the same
		C = polygonPoints.begin();	// Point[0]

		A--;						// Point[-1]
		B = polygonPoints.begin();	// Point[0]
		C++;						// Point[1]

		bool firstVertex = true;
		
		for(;C!=polygonPoints.end(); B++, C++)
		{
			Point p0 = (*A);
			Point p1 = (*B)-p0;
			Point p2 = (*C)-p0;

			if(((p1.x*p2.y)-(p2.x*p1.y))>0) 
			{
				// CCW
				concavePolygon = true;
				break;
			}


			if(firstVertex)
			{
				A = polygonPoints.begin();
				firstVertex = false;
			}
			else
			{
				A++;
			}
		}
	}
#endif

	if(cyberShape->IsConcave())
	{
		PointList polygonPoints;

		Point p;
		Box boundingBox = cyberShape->GetBoundingBox();
		double x = cyberShape->GetCadX();
		double y = cyberShape->GetCadY();
		double radians = cyberShape->GetRotation()* PI / 180.0;
		bool rotated = (fabs(radians)>0.0001);


		p.x = boundingBox.p1.x - x;
		p.y = boundingBox.p1.y - y;
		if(rotated) p.rot(radians);
		p.x += x;
		p.y += y;
		polygonPoints.push_back(p);

		p.x = boundingBox.p1.x - x;
		p.y = boundingBox.p2.y - y;
		if(rotated) p.rot(radians);
		p.x += x;
		p.y += y;
		polygonPoints.push_back(p);

		p.x = boundingBox.p2.x - x;
		p.y = boundingBox.p2.y - y;
		if(rotated) p.rot(radians);
		p.x += x;
		p.y += y;
		polygonPoints.push_back(p);

		p.x = boundingBox.p2.x - x;
		p.y = boundingBox.p1.y - y;
		if(rotated) p.rot(radians);
		p.x += x;
		p.y += y;
		polygonPoints.push_back(p);


		RenderPolygon(image, resolution, (Feature*) cyberShape, &polygonPoints, 1, grayValue, antiAlias);
	}
	else
	{
		PointList polygonPoints = cyberShape->GetPointList();
		RenderPolygon(image, resolution, (Feature*) cyberShape, &polygonPoints, 1, grayValue, antiAlias);
	}
}
	
// Create Image and Image16 instances
template void RenderCyberShape(Image& image, double resolution, CyberFeature* cyber, unsigned int grayValue, int antiAlias);
template void RenderCyberShape(Image16& image, double resolution, CyberFeature* cyber, unsigned int grayValue, int antiAlias);



//
//
// Crosses
//
//
template <typename IMAGETYPE>
void RenderCross(IMAGETYPE& image, double resolution, CrossFeature* cross, unsigned int grayValue, int antiAlias)
{
	int numPolygons = 2;
	Point point;
	PointList polygonPoints[2];
	double x, y, base, height, baseLegWidth, heightLegWidth;
	double radians = cross->GetRotation() * PI / 180.0;
	bool rotated = (fabs(radians)>0.0001);

	x = cross->GetCadX();
	y = cross->GetCadY();
	base = cross->GetSizeX();
	height = cross->GetSizeY();
	baseLegWidth = cross->GetLegSizeX();
	heightLegWidth = cross->GetLegSizeY();

	// Cross must be broken into two rectangles to draw

	// Rect0 points in CW order
	point.x = baseLegWidth / 2.0;
	point.y = -height / 2.0;
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[0].push_back(point);

	point.x = -baseLegWidth / 2.0;
	point.y = -height / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[0].push_back(point);
	
	point.x = -baseLegWidth / 2.0;
	point.y = height / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[0].push_back(point);

	point.x = baseLegWidth / 2.0;
	point.y = height / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[0].push_back(point);

	// Rect1 points in CW order
	point.x = base / 2.0;
	point.y = -heightLegWidth / 2.0;
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[1].push_back(point);

	point.x = -base / 2.0;
	point.y = -heightLegWidth / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[1].push_back(point);

	point.x = -base / 2.0;
	point.y = heightLegWidth / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[1].push_back(point);

	point.x = base / 2.0;
	point.y = heightLegWidth / 2.0; 
	if(rotated) point.rot(radians);
	point.x += x;
	point.y += y;
	polygonPoints[1].push_back(point);

	//for(PointList::iterator p=polygonPoints[1].begin(); p!=polygonPoints[1].end(); p++)
	//{
	//	G_LOG_3_SYSTEM("Cross,#%d,Point[1](Meters),%0.06lf,%0.06lf", cross->_index, p->x, p->y);
	//}

	RenderPolygon(image, resolution, (Feature*) cross, polygonPoints, numPolygons, grayValue, antiAlias);
}

// Create Image and Image16 instances
template void RenderCross(Image& image, double resolution, CrossFeature* cross, unsigned int grayValue, int antiAlias);
template void RenderCross(Image16& image, double resolution, CrossFeature* cross, unsigned int grayValue, int antiAlias);



//
//
// Diamonds
//
//
template <typename IMAGETYPE>
void RenderDiamond(IMAGETYPE& image, double resolution, DiamondFeature* diamond, unsigned int grayValue, int antiAlias)
{
	PointList polygonPoints = diamond->GetPointList();

	//for(PointList::iterator p=polygonPoints.begin(); p!=polygonPoints.end(); p++)
	//{
	//	G_LOG_3_SYSTEM("Diamond,#%d,RotatedPoint(Pixels),%0.06lf,%0.06lf", diamond->_index, p->x, p->y);
	//}

	RenderPolygon(image, resolution, (Feature*) diamond, &polygonPoints, 1, grayValue, antiAlias);
}

// Create Image and Image16 instances
template void RenderDiamond(Image& image, double resolution, DiamondFeature* diamond, unsigned int grayValue, int antiAlias);
template void RenderDiamond(Image16& image, double resolution, DiamondFeature* diamond, unsigned int grayValue, int antiAlias);

//
//
// Diamond Frame
//
//
template <typename IMAGETYPE>
void RenderDiamondFrame(IMAGETYPE& image, double resolution, DiamondFrameFeature* diamondFrame, unsigned int grayValue, int antiAlias)
{
	PointList polygonPoints = diamondFrame->GetPointList();

	RenderPolygon(image, resolution, (Feature*) diamondFrame, &polygonPoints, 1, grayValue, antiAlias);

	// Set up image to draw diamond hole
	ImgTransform trans;
	trans.Config(resolution, resolution);

	IMAGETYPE diamondHole;
	diamondHole.Configure(
		image.Columns(), 
		image.Rows(), 
		image.PixelRowStride(),
		trans,
		trans,
		true);	// create own buffer

	polygonPoints.clear();
	polygonPoints = diamondFrame->GetInnerPointList();
	RenderPolygon(diamondHole, resolution, (Feature*) diamondFrame, &polygonPoints, 1, grayValue, antiAlias);

	if(image.GetBytesPerPixel() == 1)
	{
		ClipSub(
			(unsigned char*)image.GetBuffer(), image.PixelRowStride(), 
			(unsigned char*)diamondHole.GetBuffer(), diamondHole.PixelRowStride(), 
			image.Columns(), image.Rows());
	}
	else
	{
		ClipSub(
			(unsigned short*)image.GetBuffer(), image.PixelRowStride(), 
			(unsigned short*)diamondHole.GetBuffer(), diamondHole.PixelRowStride(), 
			image.Columns(), image.Rows());
	}
}

// Create Image and Image16 instances
template void RenderDiamondFrame(Image& image, double resolution, DiamondFrameFeature* diamond, unsigned int grayValue, int antiAlias);
template void RenderDiamondFrame(Image16& image, double resolution, DiamondFrameFeature* diamond, unsigned int grayValue, int antiAlias);


//
//
// Rectangles
//
//
template <typename IMAGETYPE>
void RenderRectangle(IMAGETYPE& image, double resolution, RectangularFeature* rect, unsigned int grayValue, int antiAlias)
{
	PointList polygonPoints = rect->GetPointList();

	//for(PointList::iterator p=polygonPoints.begin(); p!=polygonPoints.end(); p++)
	//{
	//	G_LOG_3_SYSTEM("Rectangle,#%d,RotatedPoint(Meters),%0.06lf,%0.06lf", triangle->_index, p->x, p->y);
	//}

	RenderPolygon(image, resolution, (Feature*) rect, &polygonPoints, 1, grayValue, antiAlias);
}

// Create Image and Image16 instances
template void RenderRectangle(Image& image, double resolution, RectangularFeature* rect, unsigned int grayValue, int antiAlias);
template void RenderRectangle(Image16& image, double resolution, RectangularFeature* rect, unsigned int grayValue, int antiAlias);

//
//
// Rectangle Frame
//
//
template <typename IMAGETYPE>
void RenderRectangleFrame(IMAGETYPE& image, double resolution, RectangularFrameFeature* rectFrame, unsigned int grayValue, int antiAlias)
{
	PointList polygonPoints = rectFrame->GetPointList();

	RenderPolygon(image, resolution, (Feature*) rectFrame, &polygonPoints, 1, grayValue, antiAlias);

	// Set up image to draw rectanglar hole
	ImgTransform trans;
	trans.Config(resolution, resolution);

	IMAGETYPE rectHole;
	rectHole.Configure(
		image.Columns(), 
		image.Rows(), 
		image.PixelRowStride(),
		trans,
		trans,
		true);	// create own buffer

	polygonPoints.clear();
	polygonPoints = rectFrame->GetInnerPointList();
	RenderPolygon(rectHole, resolution, (Feature*) rectFrame, &polygonPoints, 1, grayValue, antiAlias);

	if(image.GetBytesPerPixel() == 1)
	{
		ClipSub(
			(unsigned char*)image.GetBuffer(), image.PixelRowStride(), 
			(unsigned char*)rectHole.GetBuffer(), rectHole.PixelRowStride(), 
			image.Columns(), image.Rows());
	}
	else
	{
		ClipSub(
			(unsigned short*)image.GetBuffer(), image.PixelRowStride(), 
			(unsigned short*)rectHole.GetBuffer(), rectHole.PixelRowStride(), 
			image.Columns(), image.Rows());
	}
}

// Create Image and Image16 instances
template void RenderRectangleFrame(Image& image, double resolution, RectangularFrameFeature* rect, unsigned int grayValue, int antiAlias);
template void RenderRectangleFrame(Image16& image, double resolution, RectangularFrameFeature* rect, unsigned int grayValue, int antiAlias);



//
//
// Triangles
//
//
template <typename IMAGETYPE>
void RenderTriangle(IMAGETYPE& image, double resolution, TriangleFeature* triangle, unsigned int grayValue, int antiAlias)
{
	PointList polygonPoints = triangle->GetPointList();

	//for(PointList::iterator p=polygonPoints.begin(); p!=polygonPoints.end(); p++)
	//{
	//	G_LOG_3_SYSTEM("Triangle,#%d,RotatedPoint(Pixels),%0.06lf,%0.06lf", triangle->_index, p->x, p->y);
	//}

	RenderPolygon(image, resolution, (Feature*) triangle, &polygonPoints, 1, grayValue, antiAlias);
}

// Create Image and Image16 instances
template void RenderTriangle(Image& image, double resolution, TriangleFeature* triangle, unsigned int grayValue, int antiAlias);
template void RenderTriangle(Image16& image, double resolution, TriangleFeature* triangle, unsigned int grayValue, int antiAlias);

//
//
// Equilateral Triangle Frame
//
//
template <typename IMAGETYPE>
void RenderTriangleFrame(IMAGETYPE& image, double resolution, EquilateralTriangleFrameFeature* triangleFrame, unsigned int grayValue, int antiAlias)
{
	PointList polygonPoints = triangleFrame->GetPointList();

	RenderPolygon(image, resolution, (Feature*) triangleFrame, &polygonPoints, 1, grayValue, antiAlias);

	// Set up image to draw triangle hole
	ImgTransform trans;
	trans.Config(resolution, resolution);

	IMAGETYPE triangleHole;
	triangleHole.Configure(
		image.Columns(), 
		image.Rows(), 
		image.PixelRowStride(),
		trans,
		trans,
		true);	// create own buffer

	polygonPoints.clear();
	polygonPoints = triangleFrame->GetInnerPointList();
	RenderPolygon(triangleHole, resolution, (Feature*) triangleFrame, &polygonPoints, 1, grayValue, antiAlias);

	if(image.GetBytesPerPixel() == 1)
	{
		ClipSub(
			(unsigned char*)image.GetBuffer(), image.PixelRowStride(), 
			(unsigned char*)triangleHole.GetBuffer(), triangleHole.PixelRowStride(), 
			image.Columns(), image.Rows());
	}
	else
	{
		ClipSub(
			(unsigned short*)image.GetBuffer(), image.PixelRowStride(), 
			(unsigned short*)triangleHole.GetBuffer(), triangleHole.PixelRowStride(), 
			image.Columns(), image.Rows());
	}
}

// Create Image and Image16 instances
template void RenderTriangleFrame(Image& image, double resolution, EquilateralTriangleFrameFeature* triangle, unsigned int grayValue, int antiAlias);
template void RenderTriangleFrame(Image16& image, double resolution, EquilateralTriangleFrameFeature* triangle, unsigned int grayValue, int antiAlias);

template <typename IMAGETYPE>
void RenderCheckerPattern(IMAGETYPE& image, double resolution, CheckerPatternFeature* checkerPattern, unsigned int grayValue, int antiAlias)
{
	int numPolygons = 2;
	PointList polygonPoints[2];
	polygonPoints[0] = checkerPattern->GetFirstPointList();
	polygonPoints[1] = checkerPattern->GetSecondPointList();

	RenderPolygon(image, resolution, (Feature*) checkerPattern, polygonPoints, numPolygons, grayValue, antiAlias);
}

// Create Image and Image16 instances
template void RenderCheckerPattern(Image& image, double resolution, CheckerPatternFeature* checkerPattern, unsigned int grayValue, int antiAlias);
template void RenderCheckerPattern(Image16& image, double resolution, CheckerPatternFeature* checkerPattern, unsigned int grayValue, int antiAlias);