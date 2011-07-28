//
// File: concreteVsWrapper.cpp
// Description: Implementation to wrap VS Find software
// Copyright (c) 2002 CyberOptics Corp.
//
// $Author: Jwaltz $
// $Revision: 30 $
// $Header: /Ghidorah/QNX6/Realtime/concreteVsWrapper.cpp 30    7/12/07 11:57a Jwaltz $
#define _USE_MATH_DEFINES // for M_PI
#include <math.h>
#include <float.h>
#include <time.h>
#include <vision.h>
#include <assert.h>

#include "concreteVsWrapper.h"
#include "errcode.h"

// uncomment the following line to print different time information
// #define __TIME_LOG

// uncomment the next line to save afid.ccf for each create_ call made
//#define __DEBUG_TEMPLATE
//#ifdef __DEBUG_TEMPLATE
//#ifdef WIN32
//extern "C" {
//#include "..\..\ANSI\Jpeg\jpeglib.h"
//}
//void write_JPEG_file (char * filename, int quality, BYTE *image_buffer, int image_height, int image_width);
//#else
//#include "CcfFile.h"
//#endif // WIN32
//#endif // __DEBUG_TEMPLATE

#ifndef __TIME_LOG

#ifndef min
template<class T> static T min(T a, T b) { return a<b ? a : b; }
#endif

#ifndef max
template<class T> static T max(T a, T b) { return a>b ? a : b; }
#endif

#else
#include "Debug.h"
#endif

//const double M_PER_PIXELS_NOM=.00002;
//const double CAMERA_WIDTH_PIXELS=1024;
//const double CAMERA_HEIGHT_PIXELS=486;

const int SKIPMARK_BACKGROUND=5;
const int SKIPMARK_EDGE=10;

struct fill_color
{
	int fore_ground;
	int back_ground;
	int hollow;
};

const char * load_template(VsEnviron oVisEnv, VsStFTemplate *ptFTemplates[], VsCamImage oCamImage, VsStFinder *Finder, const char *file_name);
const char * find_template(VsStFTemplate *ptFTemplates[], int iNumTemplates, VsCamImage oCamImage, VsStFinder *ptFinder, double time_out);
const char * lookup_finder_error(int num);
const char * lookup_template_error(int num);
static void get_differentiators(
	vs_fid_data *diff_list,
	int size,
	int fid,
	double twidth,
	double theight,
	double hwidth,
	double theta,
	int dark_to_light);

void ChangePixelSize(VsCamImage img, double m_per_pixel_x, double m_per_pixel_y)
{
	const int num_points=4;
	double meter_space[num_points][2];
	double pixel_space[num_points][2]=
	{{0,0},{1,1},{0,1},{1,0}};
	double rms, error;

	for(int i=0;i<num_points;i++)
	{
		meter_space[i][0]=pixel_space[i][0]*m_per_pixel_x;
		meter_space[i][1]=pixel_space[i][1]*m_per_pixel_y;
	}

	vsCalibrate(img, pixel_space, meter_space, num_points, &rms, &error);
}

void RotateImage(
	VsCamImage img,
	double theta    // theta is assumed to be in cycles
	)
{
	// not yet implemented or tested
#ifndef PI
   #define PI        (3.141592653589793)
#endif
#ifndef TWOPI
   #define TWOPI     (2.*PI)
#endif

	const int num_points=4;
	double meter_space[num_points][2];
	double pixel_space[num_points][2]=
	{{0,0},{1,1},{0,1},{1,0}};
	double rms, error;

	double s=sin(theta*TWOPI);
	double c=cos(theta*TWOPI);

	for(int i=0;i<num_points;i++)
	{
		meter_space[i][0] = c*pixel_space[i][0] + -s*pixel_space[i][1];
		meter_space[i][1] = s*pixel_space[i][0] + c*pixel_space[i][1];
	}

	vsCalibrate(img, pixel_space, meter_space, num_points, &rms, &error);
}


#ifdef __TIME_LOG
double print_delta(int print, const char *s, double start)
{
	if(!print) return clock();

	int c=clock();
	printf("%s %lf\n", s, (c-start)/CLOCKS_PER_SEC);
	return c;
}
#else
double print_delta(int /*print*/, const char* /*s*/, double /*start*/)
{
	return 0;
}
#endif

struct inter_help
{
private:
	double x1,x2,y1,y2;
public:
	inter_help(double c[], double width, double height)
	{
		x1=c[0]-width/2;
		x2=c[0]+width/2;
		y1=c[1]-height/2;
		y2=c[1]+height/2;
	}
	int intersected(const inter_help &i) const
	{
		return (max(x1, i.x1)<=min(x2, i.x2)) && (max(y1, i.y1)<=min(y2, i.y2));
	}
};

// return !0 if the given boxes intersect
int fiducial_intersect(
	double center1[], double width, double height,
		double center2[], double width2, double height2)
{
	inter_help one(center1, width, height);
	inter_help two(center2, width2, height2);
	return one.intersected(two);
}

const char *create_keepout(
	VsEnviron  &oVisEnv,
	VsStFTemplate   &tFTemplates,
	vs_fid_data *fid_data,			// array of fids to draw
	int num_fid_data,
	double mask_region)
{
	const char *ret=0;

	//VsEnviron oVisEnv = vswrapper::getEnv();

	VsCamImage mask_image=vsCreateCamImageFromBuffer( oVisEnv, 0, tFTemplates.pbMask, (int)tFTemplates.tPixelRect.dWidth, (int)tFTemplates.tPixelRect.dHeight,
		(int)tFTemplates.tPixelRect.dWidth, VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, 1);

	double mask_center[2];
	mask_center[0]=tFTemplates.tPixelRect.dWidth/2.0;
	mask_center[1]=tFTemplates.tPixelRect.dHeight/2.0;

	// not ready for donut yet
	// no real test for other shapes yet
	// polygons are not covered at this point or planed for current releases
	for(int i=0;i<num_fid_data&&!ecfail(ret)&&mask_region>0;i++)
	{
		int fore_ground=1;
		int back_ground=0;

		double keep_out_width=fid_data[i].twidth*mask_region;
		double keep_out_height=fid_data[i].theight*mask_region;
		double keep_out_hollow=fid_data[i].hwidth*mask_region;

		if(keep_out_width>fid_data[i].twidth)
			ret="600 Fiducial mask larger than template width";
		else if(keep_out_height>fid_data[i].theight)
			ret="600 Fiducial mask larger than template height";
		else /*if(keep_out_hollow>fid_data[i].hwidth && 0)  // ajrajr not yet ready for non donut
			ret="600 Fiducial mask larger than template size";
		else*/ if(vsDrawFiducial(mask_image, fid_data[i].fid, mask_center,
			keep_out_width, keep_out_height, keep_out_hollow, fore_ground,
				back_ground, back_ground, !i, i)==-1)
					ret= "666 Failed to draw fiducial mask";
	}

#ifdef __DEBUG_TEMPLATE
#ifdef WIN32
	write_JPEG_file ("mask.jpg", 100, tFTemplates.pbMask, tFTemplates.tPixelRect.dHeight, tFTemplates.tPixelRect.dWidth);
#else
	write_image(tFTemplates.pbMask, tFTemplates.tPixelRect.dWidth, tFTemplates.tPixelRect.dHeight, 0, 0, "mask.ccf", 0, 0, 0, 1, 1, "help me");	
#endif // WIN32
#endif
	vsDispose(mask_image);

	//vswrapper::releaseEnv();

	return ret;
}

// ImageInOut = clip(ImageInOut-Image2+iGreyOffset)
void ImageSubClip(
	unsigned char* pbBufInOut,
	unsigned int iSpanInOut, 
	unsigned char* pbBuf2,
	unsigned int iSpan2,
	unsigned int iWidth,
	unsigned int iHeight, 
	int iGreyOffset=0)
{
	unsigned char* inOutLine = pbBufInOut;
	unsigned char* inLine = pbBuf2;
	for(unsigned int iy=0; iy<iHeight; iy++)
	{
		for(unsigned int ix=0; ix<iWidth; ix++)
		{
			int iTemp = (int)inOutLine[ix]-(int)inLine[ix] + iGreyOffset;
			if(iTemp<0) iTemp = 0;
			if(iTemp>255) iTemp = 255;
			inOutLine[ix] = (unsigned char)iTemp;
		}
		inOutLine += iSpanInOut;
		inLine  += iSpan2;
	}
}

// ImageInOut = clip(ImageInOut-Image2+iGreyOffset)
bool ImageClipSub(VsCamImage oImgInOut, const VsCamImage oImg2, int iGreyOffset=0)
{
	VsStCamImageInfo tCInfo1;
	vsInqCamImageInfo(oImgInOut, &tCInfo1);
	unsigned int iSpan1 = tCInfo1.iBufSpan;
	unsigned int iWidth1 = tCInfo1.iBufWidth;
	unsigned int iHeight1 = tCInfo1.iBufHeight;
	unsigned char* pbBuf1 = tCInfo1.pbBuf;

	VsStCamImageInfo tCInfo2;
	vsInqCamImageInfo(oImg2, &tCInfo2);
	unsigned int iSpan2 = tCInfo2.iBufSpan;
	unsigned int iWidth2 = tCInfo2.iBufWidth;
	unsigned int iHeight2 = tCInfo2.iBufHeight;
	unsigned char* pbBuf2 = tCInfo2.pbBuf;

	if((iWidth1 != iWidth2) ||
		(iHeight1 != iHeight2))
		return(false);

	ImageSubClip(
		pbBuf1,
		iSpan1, 
		pbBuf2,
		iSpan2,
		iWidth1,
		iHeight1,
		iGreyOffset);

	return(true);
}

concreteVsWrapper::concreteVsWrapper() :
	vswrapper( this ),
	m_pixel_size(-1),
	m_width_pixels(-1),
	m_height_pixels(-1),
	pFTemplate_(0)
{
	m_iCurrentNodeID = 0;
	m_hMutex = CreateMutex(NULL, FALSE, NULL);		// Initially not owned  
}

concreteVsWrapper::~concreteVsWrapper()
{
	if(pFTemplate_) 
		delete pFTemplate_;

	CloseHandle(m_hMutex);
}
/*
// This function will only draw one fiducial per template
const char *concreteVsWrapper::create_and_save(
	const char *template_name,		// name of template
	templatetype tpl,				// Type of template to generate
	vs_fid_data *fid_data,			// array of fids to draw
	int num_fid_data,				// number of elements in the fid array
	vs_fid_poly_data *poly_data,	// polygon to draw
	double *min_scale,				// min scale array (x,y) to search for (0-?) =< max_scale, 1 is unity
	double *max_scale,				// max scale array (x,y) to search for (0-?) >= min_scale, 1 is unity
	double low_accept,				// Minimum score to accept in the Low Resolution image
	double high_accept,				// Minimum score to accept in the High Resolution image
	int iDepth,
	vs_fid_data *diff_list//=0,
	int diff_size//=0,
	double mask_region//=0			// creates masked region that vsfind ignores. No effect if <=0
	)
{
	assert( m_width_pixels > 0 && m_height_pixels > 0);
	if( m_width_pixels <= 0 || m_width_pixels <= 0 )
		return "600 invalid image size";
	
	int im_width = 0;
	int im_height = 0;
	for(int i = 0; i < diff_size; i++)
	{
		im_width += (int)(diff_list[i].twidth * 2);
		im_height += (int)(diff_list[i].theight * 2);
	}
	int width = 1;
	int height = 1;
	while(im_width >= width) width <<= 1;
	while(im_height >= height) height <<= 1;
	if(tpl == SKIPMARK || width == 1 || height == 1)
	{
		width = (int)m_width_pixels;
		height = (int)m_height_pixels;
	}

	VsEnviron oVisEnv = vswrapper::getEnv();

	VsCamImage cam_image=vsCreateCamImage( oVisEnv, 0, 0, (int)width, (int)height,
		VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, 1);
	VsStFTemplate   tFTemplates;
	VsStFTemplate   *ptFTemplates[]={&tFTemplates};
	VsStToolRect tool;
	const double dMinAngle=0;
	const double dMaxAngle=0;

	tool.dCenter[0]  = (width)/2.0;
	tool.dCenter[1]  = (height)/2.0;
	tool.dAngle	  = 0.0;

	fill_color fill_colors[2]={{65 , 192, 192}, {192 , 65, 65}};

	if(tpl == SKIPMARK)
	{
		fill_colors[0].fore_ground=128;
		fill_colors[0].back_ground=192;
		fill_colors[0].hollow=64;

		fill_colors[1].fore_ground=128;
		fill_colors[1].back_ground=64;
		fill_colors[1].hollow=192;
	}

	const char *ret=0;
	int fore_ground=0, back_ground=0, hollow=0;

	double start=clock();

	double twidth = 0.0;
	double theight = 0.0;

	for(int i=0;i<num_fid_data&&!ecfail(ret);i++)
	{
		fore_ground=fill_colors[!!fid_data[i].dark_to_light].fore_ground;
		back_ground=fill_colors[!!fid_data[i].dark_to_light].back_ground;
		hollow=fill_colors[!!fid_data[i].dark_to_light].hollow;

		twidth=max(twidth, fid_data[i].twidth);
		theight=max(theight, fid_data[i].theight);

		if(vsDrawFiducial(cam_image, fid_data[i].fid, tool.dCenter,
			fid_data[i].twidth, fid_data[i].theight, fid_data[i].hwidth, fore_ground,
				back_ground, hollow, !i, i)==-1)
		{
			if(tpl == SKIPMARK)
				ret= "666 Failed to draw skipmark template";
			else
				ret= "666 Failed to draw fiducial template";
		}
	}
	start=print_delta(1, "draw", start);

	if(!num_fid_data)
	{
		twidth=poly_data->width;
		theight=poly_data->height;

		fore_ground=fill_colors[!!poly_data->dark_to_light].fore_ground;
		back_ground=fill_colors[!!poly_data->dark_to_light].back_ground;

		vector3 p0 = poly_data->poly.points().shift();
		vector3 p1 = poly_data->poly.points().shift();
		vector3 p2 = poly_data->poly.points().shift();

		double points[4][2] = {{p0.x,p0.y}, {p2.x,p2.y}, {p1.x,p1.y}, {p0.x,p0.y}};

		if(vsDrawConvexPolygon(cam_image, points, 3, fore_ground, back_ground, TRUE, FALSE) == -1)
		{
			if(tpl == SKIPMARK)
				ret = "666 Failed to draw Polygon skipmark template";
			else
				ret = "666 Failed to draw Polygon fiducial template";
		}
	}

	if(tpl == SKIPMARK)
	{
		// Skipmarks require much less background than fiducials
		// this is because we are less concerned with edges
		tool.dWidth	= twidth  + 2*SKIPMARK_BACKGROUND;
		tool.dHeight = theight + 2*SKIPMARK_BACKGROUND;
	}
	else
	{
		tool.dWidth	= twidth  * 1.4; // add a little background to the template.
		tool.dHeight = theight * 1.4; // This is a must.
	}

	if(ecfail(ret))
	{
	}
	else if (vsCreateFTemplate (cam_image, &tool, iDepth,
		dMinAngle, dMaxAngle, min_scale, max_scale, &tFTemplates) == -1)
			ret=lookup_template_error(tFTemplates.iResultFlags); // fail
	else if(ecfail(ret=create_keepout(oVisEnv, tFTemplates, fid_data, num_fid_data, mask_region)))
	{
	}
	else
	{
		start=print_delta(1, "create template", start);
		// Set some of the template properties - even though most of them
		//are the default values already set by vsCreateFTemplate()
		tFTemplates.iPyramidType = VS_FINDER_PYRAMID_AVERAGE;
		tFTemplates.yUniformScaling = TRUE;
		tFTemplates.iSpeed = 90; //-1; // auto set 
		tFTemplates.iTemplateType = VS_FINDER_GRAY_SCALE_BASED;
		tFTemplates.yMultiLayerBuildup = FALSE;
		tFTemplates.iAccuracy = 50;
		tFTemplates.iIgnoreValuesAbove = 255; // default
		tFTemplates.iIgnoreValuesBelow = 0;
		tFTemplates.iCorrelationType = 1; // gain and offset
		tFTemplates.dGainFactor = 20.0;
		tFTemplates.iOffsetValue = 255; // ignored if gain only
		tFTemplates.yAllowNegatives = FALSE; // ignored in gain only 
		tFTemplates.yAllowPyramidTypeChange = 0;
		tFTemplates.yAllowCorrelationTypeChange = FALSE ;
		tFTemplates.iMinimumPyramidDepth = 1;
		tFTemplates.dLoResMinScore     = 0.50;
		tFTemplates.dHiResMinScore     = 0.50;
		tFTemplates.iMaxResultPoints   = 2; // # matches to find 
		//tFTemplates.yComputeTrueNgcScore = TRUE;

		// Setting the training space containing other fiducials or objects of interest that the finder
		tool.dWidth = width;
		tool.dHeight = height;
		tool.dAngle	= 0;

		VsStFinder Finder;

		int half_diff=(int)((diff_size/2.0)+.5);
		double quarter_height=height/4.0;

		double diff_center[2]={5,quarter_height};
		int the_one=0;
		for(int diff=0;diff<diff_size;diff++)
		{
			if(diff==half_diff)
			{
				diff_center[1]=quarter_height*3;
				diff_center[0]=5;
			}

			if(diff_center[1]-diff_list[the_one].theight/2<=0)
				continue;
			if(diff_center[1]+diff_list[the_one].theight/2>=height)
				continue;

			diff_center[0]+=diff_list[the_one].twidth/2;

			if(fiducial_intersect(diff_center, diff_list[the_one].twidth, diff_list[the_one].theight,
				tool.dCenter, twidth, theight))
					diff_center[0]=tool.dCenter[0]+twidth+diff_list[the_one].twidth/2;

			if(diff_center[0]+diff_list[the_one].twidth/2.>=width)
				continue;

			if(vsDrawFiducial(cam_image, diff_list[the_one].fid, diff_center,
				diff_list[the_one].twidth, diff_list[diff].theight, diff_list[the_one].hwidth, fore_ground,
					back_ground, back_ground, 0, 1)==-1)
						ret= "666 Failed to draw difference shape";

			diff_center[0]+=diff_list[the_one].twidth/2+30;
			the_one++;
		}

#ifdef __DEBUG_TEMPLATE
// used for debug to save the image out to a file
// ajrajr
		VsStCamImageInfo tData;
		vsInqCamImageInfo(cam_image, &tData);
#ifdef WIN32
		write_JPEG_file ("afid.jpg", 100, tData.pbBuf, height, width);
#else
		write_image(tData.pbBuf, width, height, 0, 0, "afid.ccf", 0, 0, 0, 1, 1, "help me");
#endif // WIN32

#endif

		if (vsCreateFinder(cam_image, &tool, 1, ptFTemplates, &Finder) == -1)
			ret= lookup_finder_error(Finder.iResultFlags);
		else
		{
			tFTemplates.dLoResMinScore     = low_accept;//0.50;
			tFTemplates.dHiResMinScore     = high_accept;//0.60;

			if (vsSaveTemplate(&tFTemplates, (char *)template_name) != VS_OK)
			{
				vsDispose((void *)&Finder);
				ret= lookup_template_error(tFTemplates.iResultFlags);
			}
			else
			{
				start=print_delta(1, "save template", start);
				ret= 0;
				vsDispose((void *)&Finder);
			}
		}

		vsDispose((void *)&tFTemplates);
	}

	vsDispose(cam_image);

	vswrapper::releaseEnv();

	return ret;
}

const char *concreteVsWrapper::create_and_save(const char *template_name, templatetype tpl, int fid, double twidth,
							double theight, double hwidth, double theta, int dark_to_light,
							double *min_scale, double *max_scale, double low_accept,
							double high_accept,	int depth, double mask_region)
{
	vs_fid_data fid_data;
	const int diff_size=5;
	vs_fid_data diff_list[diff_size];

	fid_data.fid=fid;
	fid_data.twidth=twidth;
	fid_data.theight=theight;
	fid_data.hwidth=hwidth;
	fid_data.theta=theta;
	fid_data.dark_to_light=dark_to_light;

	if(fabs(theta) > 0.0) 
	{
		fid_data.twidth = theight;
		fid_data.theight = twidth;
	}

	if(tpl == FIDUCIAL)
		get_differentiators(diff_list, diff_size, fid, twidth, theight,
			hwidth, theta, dark_to_light);

	return create_and_save(template_name, tpl, &fid_data, 1, 0, min_scale, max_scale, low_accept,
		high_accept, depth, diff_list, diff_size, mask_region);
}

const char * concreteVsWrapper::create_disc_template(const char *file_name, templatetype tpl, double r,
			  double theta, int dark_to_light, double *min_scale, double *max_scale,
			  double low_accept, double high_accept, double mask_region, int depth)
{
	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0)
	{
		r/=m_pixel_size;
		double twidth=r*2, theight=r*2;

		if(tpl == SKIPMARK)
			return create_and_save(file_name, tpl, CIRCLE_H_FIDUCIAL, twidth+2*SKIPMARK_EDGE,
				theight+2*SKIPMARK_EDGE, SKIPMARK_EDGE, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
		else
			return create_and_save(file_name, tpl, CIRCLE_FIDUCIAL, twidth, theight, 0, theta,
				dark_to_light, min_scale, max_scale, low_accept, high_accept, depth, mask_region);
	}
	else
	{
		return "600 bad pixel size";
	}
}

const char * concreteVsWrapper::create_rectangle_template(const char *file_name, templatetype tpl,
			   double base, double height, double theta, int dark_to_light,
			   double *min_scale, double *max_scale, double low_accept,
			   double high_accept, double mask_region, int depth)
{
	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		base/=m_pixel_size;
		height/=m_pixel_size;

		if(tpl == SKIPMARK)
			return create_and_save(file_name, tpl, SQUARE_H_FIDUCIAL, base+2*SKIPMARK_EDGE,
				height+2*SKIPMARK_EDGE, SKIPMARK_EDGE, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
		else
			return create_and_save(file_name, tpl, SQUARE_FIDUCIAL, base,
				height, 0, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_diamond_template(const char *file_name, templatetype tpl,
			   double base, double height, double theta, int dark_to_light,
			   double *min_scale, double *max_scale, double low_accept,
			   double high_accept, double mask_region, int depth)
{
	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		base/=m_pixel_size;
		height/=m_pixel_size;

		if(tpl == SKIPMARK)
			return create_and_save(file_name, tpl, DIAMOND_H_FIDUCIAL, base+2*SKIPMARK_EDGE,
				height+2*SKIPMARK_EDGE, SKIPMARK_EDGE, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
		else
			return create_and_save(file_name, tpl, DIAMOND_FIDUCIAL, base,
				height, 0, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_triangle_template(const char *file_name, templatetype tpl,
			  double base, double height, double offset, double theta,
			  int dark_to_light, double *min_scale,  double *max_scale,
			  double low_accept, double high_accept, double mask_region, int depth)
{
	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		vs_fid_poly_data poly_data;

		// FOV width and height.
		double wdth=m_width_pixels/2.0;
		double hgt=m_height_pixels/2.0;

		// Convert from meters to pixels.
		base/=m_pixel_size;
		height/=m_pixel_size;
		offset/=m_pixel_size; // offset to x of top point of triangle

		// Create rotation matrix from theta
		matrix3 rotation = matrix3().rotate(2, (2 * M_PI * -theta));

		polygon triangle;

		// FOV Center
		vector3 fov_center=vector3(wdth,hgt);

		//Center of triange bounding box.
		vector3 bound_center = rotation * vector3(base/2.0,height/2.0);

		triangle.points().push(vector3(0,0));
		triangle.points().push(vector3(base, 0));
		triangle.points().push(vector3(offset, height));
		triangle *= rotation;
		rect tri_bound = triangle.bound();
		base = tri_bound.width();
		height = tri_bound.height();

		vector3 tri_cent = triangle.centroid();
		vector3 delta = bound_center - tri_cent;

		// move centroid of triangle to lower left corner
		triangle-=triangle.centroid();

		// move centroid of triangle to fov center.
		triangle+=fov_center;

		// get new triangle centroid.
		vector3 centroid = 	triangle.centroid();

		// Now triangle centroid is centered in fov.
		// Need to ensure template is big enough.
		// Delta is difference between triangle bounding box and centrod.
		// So base would be enough if centered, but we need extra delta
		// on at least once side and both gives some border.
		poly_data.width = base + (2 * fabs(delta.x));
		poly_data.height = height + (2 * fabs(delta.y));
		poly_data.poly = triangle;
		poly_data.dark_to_light = dark_to_light;

		return create_and_save(file_name, tpl, 0, 0, &poly_data, min_scale, max_scale,
			low_accept, high_accept, depth, 0, 0);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_donut_template(const char *file_name, templatetype tpl,
			   double inner_radius, double outer_radius, double theta,
			   int dark_to_light, double *min_scale, double *max_scale,
			   double low_accept, double high_accept, double mask_region,
			   int depth)
{
	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		inner_radius/=m_pixel_size;
		outer_radius/=m_pixel_size;

		double twidth=outer_radius*2, theight=outer_radius*2;

		return create_and_save(file_name, tpl, CIRCLE_H_FIDUCIAL, twidth, theight,
			outer_radius-inner_radius, theta, dark_to_light, min_scale, max_scale,
			low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_cross_template(const char *file_name, templatetype tpl,
			   double base, double height, double base_leg, double height_leg,
			   double theta, int dark_to_light, double *min_scale, double *max_scale,
			   int rounded_edges, double low_accept, double high_accept,
			   double mask_region, int depth)
{
	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		if(rounded_edges != 0)
		{
			base/=m_pixel_size;
			height/=m_pixel_size;

			if(fabs(base_leg-height_leg) > DBL_EPSILON)
				return "500 Rounded-edge Cross Fiducials with different width legs are not valid";

			base_leg/=m_pixel_size;

			return create_and_save(file_name, tpl, PLUS_FIDUCIAL_ROUNDED, base, height, base_leg,
				theta, dark_to_light, min_scale, max_scale, low_accept, high_accept, depth,
				mask_region);
		}
		else
		{
			vs_fid_data fid_data[2];

			base/=m_pixel_size;
			height/=m_pixel_size;
			height_leg/=m_pixel_size;
			base_leg/=m_pixel_size;

			fid_data[0].fid=SQUARE_FIDUCIAL;
			fid_data[0].twidth=base;
			fid_data[0].theight=base_leg;
			fid_data[0].hwidth=0;
			fid_data[0].theta=theta;
			fid_data[0].dark_to_light=dark_to_light;

			fid_data[1].fid=SQUARE_FIDUCIAL;
			fid_data[1].twidth=height_leg;
			fid_data[1].theight=height;
			fid_data[1].hwidth=0;
			fid_data[1].theta=theta;
			fid_data[1].dark_to_light=dark_to_light;

			const int diff_size=5;
			vs_fid_data diff_list[diff_size];

			get_differentiators(diff_list, diff_size, PLUS_FIDUCIAL, base, height,
				0, theta, dark_to_light);

			return create_and_save(file_name, tpl, fid_data, 2, 0, min_scale, max_scale,
				low_accept, high_accept, depth, diff_list, diff_size, mask_region);
		}
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::load_and_find(const char *file_name, unsigned char *image, int width, int height,
						   double &x, double &y, double &correlation, double &ambig, double *ngc,
						   double search_center_x, double search_center_y, double search_width,
						   double search_height, double time_out, int y_origin_ll, double min_accept,
						   double max_accept, int num_finds)
{
	double start=clock();

	VsEnviron oVisEnv = vswrapper::getEnv();
	VsStFinder Finder;
	VsStFTemplate   tFTemplates, *ptFTemplates[]={&tFTemplates};
	VsCamImage cam_image=vsCreateCamImageFromBuffer( oVisEnv, 0, image, width, height,
		width, VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, y_origin_ll);

	if(ngc)
		tFTemplates.yComputeTrueNgcScore = TRUE;

	start=print_delta(1, "alloc vs stuff", start);
	const char *res=load_template(oVisEnv, ptFTemplates, cam_image, &Finder, file_name);
	start=print_delta(1, "loading", start);

	if(ecfail(res))
	{
		y=x=0;
		correlation=0;
		ambig=1;
		if(ngc) *ngc=0;
	}
	else
	{
		// set the search space for vsFind()
		Finder.tToolRect.dCenter[0]  =  search_center_x;
		Finder.tToolRect.dCenter[1]  = search_center_y;
		Finder.tToolRect.dWidth	  = search_width;
		Finder.tToolRect.dHeight	  = search_height;
		Finder.tToolRect.dAngle	  = 0.0;

		if(min_accept>0) tFTemplates.dLoResMinScore=min_accept;
		if(max_accept>0) tFTemplates.dHiResMinScore=max_accept;
		if(num_finds>0) tFTemplates.iMaxResultPoints=num_finds;

		res=find_template(ptFTemplates, 1, cam_image, &Finder, time_out);

		start=print_delta(1, "found", start);

		if(tFTemplates.ptFPoint && tFTemplates.ptFPoint[0].dScore!=-1)
		{
			x=tFTemplates.ptFPoint[0].dLoc[0];
			y=tFTemplates.ptFPoint[0].dLoc[1];
			correlation=tFTemplates.ptFPoint[0].dScore;
			if(tFTemplates.iNumResultPoints>1)
				ambig=tFTemplates.ptFPoint[1].dScore/correlation;
			else
				ambig=0;

			if(ngc)
				*ngc=tFTemplates.ptFPoint[0].dNgcScore;
		}
		else
		{
			y=x=0;
			correlation=0;
			ambig=1;
			if(ngc) *ngc=0;
		}

		vsDispose(&Finder);
		vsDispose(&tFTemplates);
	}

	vsDispose(cam_image);
//	vsDispose(oVisEnv);

	print_delta(1, "vsDisposes", start);

	vswrapper::releaseEnv();

	return res;
}


// loads template from file
	int concreteVsWrapper::load_finder_template(
		const char *pcFileName        // name of the template file
	)
{
	assert( vsAttachThread() != -1 );

	int iReturn=0;
	// load finder template

	if( pFTemplate_ )
		delete pFTemplate_;
	pFTemplate_ = new VsStFTemplate;
	char* buf = new char[strlen(pcFileName)+1];

//	assert( vsAttachThread() != -1 );
	if( pFTemplate_ && buf )
	{
		strcpy( buf, pcFileName );
		iReturn = vsLoadFTemplate(NULL, (VsStFTemplate *)pFTemplate_, buf);
	}
	delete[] buf;

	//if (iReturn == VS_OK) {
	//	OK_PRINT(((VsStFTemplate *) ptFTemplate));
	//	return(VS_OK);
	//}
	//else {
	//	ERR_PRINT(((VsStFTemplate *) ptFTemplate));
	//}

	assert( vsDetachThread( NULL ) != -1 );

	return iReturn;
}

// performs vsfind given template object.
const char * concreteVsWrapper::find_from_template(
	unsigned char *image,   // buffer containing the image
	int width,              // width of the image in pixels
	int height,             // height of the image in pixels
	double &x,              // returned x location of the center of the template from the origin
	double &y,              // returned x location of the center of the template from the origin
	double &correlation,    // match score 0-1
	double &ambig,          // ratio of (second best/best match) score 0-1
	double *ngc,			// Normalized Grayscale Correlation Score 0-1
	double search_center_x, // x center of the search region in pixels
	double search_center_y, // y center of the search region in pixels
	double search_width,    // width of the search region in pixels
	double search_height,   // height of the search region in pixels
	double time_out,        // number of seconds to search maximum. If limit is reached before any results found, an error will be generated
	int y_origin_ll,      // If !0 origin in start of the image is 0,0 increasing in Y and X as you move forward
	double min_accept,   // If >0 minimum score to persue at min pyramid level to look for peak override
	double max_accept,   // If >0 minumum score to accept at max pyramid level to look for peak override
	int num_finds        // If > 0 number of versions of the template to find
)
{

	double start=clock();
	VsStFinder Finder;

	//VsStFTemplate   tFTemplates, *ptFTemplates[]={&tFTemplates};

	VsEnviron oVisEnv = vswrapper::getEnv();

	VsCamImage cam_image=vsCreateCamImageFromBuffer( oVisEnv, 0, image, width, height,
		width, VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, y_origin_ll);

	if(ngc)
		pFTemplate_->yComputeTrueNgcScore = TRUE;

	start=print_delta(1, "alloc vs stuff", start);

	const char *res=NULL;
	
	// CHECK FOR VALID IMAGE.

	VsStToolRect tToolRect;

	VsStCamImageInfo ci;
	vsInqCamImageInfo(cam_image, &ci);

	double cam_width = (double) ci.iBufWidth;
	double cam_height = (double) ci.iBufHeight;

	tToolRect.dCenter[0] = (cam_width-1) * 0.5;
	tToolRect.dCenter[1] = (cam_height-1) * 0.5;
	tToolRect.dWidth	  = cam_width;
	tToolRect.dHeight	  = cam_height;
	tToolRect.dAngle	  = 0;

	if(vsCreateFinder(cam_image, &tToolRect, 1, &pFTemplate_, &Finder) == -1)
		res = lookup_finder_error(Finder.iResultFlags);



	start=print_delta(1, "loading", start);

	if(ecfail(res))
	{
		y=x=0;
		correlation=0;
		ambig=1;
		if(ngc) *ngc=0;
	}
	else
	{
		Finder.tToolRect.dCenter[0]  =  search_center_x;
		Finder.tToolRect.dCenter[1]  = search_center_y;
		Finder.tToolRect.dWidth	  = search_width;
		Finder.tToolRect.dHeight	  = search_height;
		Finder.tToolRect.dAngle	  = 0.0;

		if(min_accept>0) pFTemplate_->dLoResMinScore=min_accept;
		if(max_accept>0) pFTemplate_->dHiResMinScore=max_accept;
		if(num_finds>0) pFTemplate_->iMaxResultPoints=num_finds;

		res=find_template(&pFTemplate_, 1, cam_image, &Finder, time_out);

		start=print_delta(1, "found", start);

		if(pFTemplate_->ptFPoint && pFTemplate_->ptFPoint[0].dScore!=-1)
		{
			x=pFTemplate_->ptFPoint[0].dLoc[0];
			y=pFTemplate_->ptFPoint[0].dLoc[1];
			correlation=pFTemplate_->ptFPoint[0].dScore;
			if(pFTemplate_->iNumResultPoints>1)
				ambig=pFTemplate_->ptFPoint[1].dScore/correlation;
			else
				ambig=0;

			if(ngc)
				*ngc=pFTemplate_->ptFPoint[0].dNgcScore;
		}
		else
		{
			y=x=0;
			correlation=0;
			ambig=1;
			if(ngc) *ngc=0;
		}

		vsDispose(&Finder);
		//vsDispose(pFTemplate_);
	}

	vsDispose(cam_image);
//	vsDispose(oVisEnv);

	vswrapper::releaseEnv();

	print_delta(1, "vsDisposes", start);

	return res;
}
*/


struct vserror
{
	int code;
	const char *str;
};

const char * lookup_finder_error(int num)
{
	const int NUM_VS_ERRORS=24;

	const vserror finder_vserrors[NUM_VS_ERRORS]=
	{
		{VS_FINDER_ALLOC_FAILED, "570 Failed to allocate memory"},
		{VS_FINDER_ANGLES_SCALE_RESET, "600 Finder angle scales reset"},
		{VS_FINDER_ACCURACY_OVR_SPECIFIED, "600 Finder Accuracy over specified"},
		{VS_FINDER_DIFF_TEMPLATE_GAIN_OFFSET, "600 Different template gain offset"},
		{VS_FINDER_DIFF_TEMPLATE_TYPES, "600 Different template types"},
		{VS_FINDER_FORCED_SLOW_LEARNING, "600 Forced slow learning"},
		{VS_FINDER_INVALID_ACQUISITION, "500 Invalid acquisition"},
		{VS_FINDER_INVALID_CAM_IMAGE, "500 Invalid camera image"},
		{VS_FINDER_INVALID_NUMBER_OF_TEMPLATES, "600 Invalid number of templates"},
		{VS_FINDER_INVALID_OBJECT, "600 Invalid object"},
		{VS_FINDER_INVALID_TEMPLATE, "500 Invalid template"},
		{VS_FINDER_INVALID_TEMPLATE_ACCURACY, "600 Invalid template accuracy"},
		{VS_FINDER_INVALID_TEMPLATE_GAIN_OFFSET, "600 Invalid template gain offset"},
		{VS_FINDER_INVALID_TEMPLATE_SPEED, "600 Invalid template speed"},
		{VS_FINDER_NO_LICENSE, "600 No valid VS Find license"},
		{VS_FINDER_OFF_AXIS_ANGLE,"600 Off axis angle"},
		{VS_FINDER_PYRAMID_DEPTH_CLIPPED, "600 Pyramid depth clipped"},
		{VS_FINDER_PYRAMID_STRATEGY_RESET, "600 Pyramid strategy resset"},
		{VS_FINDER_TEMPLATE_OUTSIDE_ROI, "600 Finder outside of Region of Interest"},
		{VS_FINDER_TIMED_OUT, "600 Finder timed out"},
		{VS_FINDER_TOOL_CLIPPED_BY_IMAGE, "500 Tool clipped by the image"},
		{VS_FINDER_TOOL_OUTSIDE_IMAGE, "500 Tool located outside of the image"},
		{VS_FINDER_TRAINING_ABORTED, "600 Finder training aborted"},
		{VS_FINDER_TRAINING_FAILED, "600 Finder training faliled"}
	};

	for(int i=0;i<NUM_VS_ERRORS;i++)
		if(finder_vserrors[i].code==num)
			return finder_vserrors[i].str;

	return "600 Unknown Finder Error";

}
const char * lookup_template_error(int num)
{
	const int NUM_VS_ERRORS=18;
	const vserror template_vserrors[NUM_VS_ERRORS]=
	{
		{VS_FTEMPLATE_ALLOC_FAILED, "570 Failed to allocate memory"},
		{VS_FTEMPLATE_DELETE_FAILED, "570 Failed to delete file"},
		{VS_FTEMPLATE_FINDER_FAILED, "570 Failed to find"},
		{VS_FTEMPLATE_INVALID_ACQUISITION, "570 Acquition failed"},
		{VS_FTEMPLATE_INVALID_ANGLES, "600 Invalid angles"},
		{VS_FTEMPLATE_INVALID_CAM_IMAGE, "600 Invalid camera image"},
		{VS_FTEMPLATE_INVALID_DEPTH, "600 Invalid depth"},
		{VS_FTEMPLATE_INVALID_SCALE_FACTORS, "600 Invalid scale factors"},
		{VS_FTEMPLATE_INVALID_TEMPLATE, "600 Invalid tempate"},
		{VS_FTEMPLATE_IO_FAILED, "570 Failed file io"},
		{VS_FTEMPLATE_NO_LICENSE, "600 No valid VS Find license"},
		{VS_FTEMPLATE_NOT_TRAINED, "500 The template was not trained"},
		{VS_FTEMPLATE_OFF_AXIS_ANGLE,"600 Off axis angle"},
		{VS_FTEMPLATE_OPEN_FAILED, "570 Failed to open file"},
		{VS_FTEMPLATE_TOOL_CLIPPED_BY_IMAGE, "500 Tool clipped by the image"},
		{VS_FTEMPLATE_TOOL_OUTSIDE_IMAGE, "500 Tool located outside of the image"},
		{VS_FTEMPLATE_TOO_SMALL, "500 Template is to small"},
		{VS_FTEMPLATE_UNIFORM_TEMPLATE, "600 Uniform template"}
	};

	for(int i=0;i<NUM_VS_ERRORS;i++)
		if(template_vserrors[i].code==num)
			return template_vserrors[i].str;

	return "600 Unknown Template Error";
}

const char * load_template(VsEnviron /*oVisEnv*/, VsStFTemplate *ptFTemplates[], VsCamImage oCamImage, VsStFinder *ptFinder, const char *file_name)
{
	BOOL bComputeNgcScore = ptFTemplates[0]->yComputeTrueNgcScore;

	VsEnviron oVisEnv = vswrapper::getEnv();

	if (vsLoadTemplate(oVisEnv, (char *)file_name, VS_FINDER, 0, 0, 0, ptFTemplates[0]) != VS_OK)
		return lookup_template_error(ptFTemplates[0]->iResultFlags);

	if (bComputeNgcScore)
		ptFTemplates[0]->yComputeTrueNgcScore = TRUE;

	/* Loading a finder template is similar to creating a finder
	   template, except that all training information are also loaded.
	   However, remember a finder structure needs to be initialized
     properly to run the finder. So, vsCreateFinder() must be called */

	VsStToolRect tToolRect;

	VsStCamImageInfo ci;
	vsInqCamImageInfo(oCamImage, &ci);

	double cam_width = (double) ci.iBufWidth;
	double cam_height = (double) ci.iBufHeight;

	tToolRect.dCenter[0] = (cam_width-1) * 0.5;
	tToolRect.dCenter[1] = (cam_height-1) * 0.5;
	tToolRect.dWidth	  = cam_width;
	tToolRect.dHeight	  = cam_height;
	tToolRect.dAngle	  = 0;

	if(vsCreateFinder(oCamImage, &tToolRect, 1, ptFTemplates, ptFinder) == -1)
		return lookup_finder_error(ptFinder->iResultFlags);

	vswrapper::releaseEnv();

	return 0;
}

const char * find_template(VsStFTemplate *ptFTemplates[], int iNumTemplates, VsCamImage oCamImage, VsStFinder *ptFinder, double time_out)
{
	/* Now, find the objects using vsFind() */
	ptFinder->eDisplayMode = VS_DRAW_NO_GRAPHICS;
	ptFinder->yFirstTime	 = TRUE;
	ptFinder->dTimeout	 = time_out;

	if (vsFind(oCamImage, iNumTemplates, ptFTemplates, ptFinder) == -1)
	{
		// decrease the template pyramid height by 1
		--ptFTemplates[0]->iDepth;
		if (ptFTemplates[0]->iDepth == 0)
		{
			printf("\n Error in Finding Objects...%d", ptFinder->iResultFlags);
			//ajrajr goto END;
		}
	}

#if 0
	printf("\nFinder Time = %g ms\n", ptFinder->dRecogTime);
	for (int iTNum = 0; iTNum < iNumTemplates; iTNum++)
	{
		printf("Number of Matches: %d\n", (*ptFTemplates)[iTNum].iNumResultPoints);
		for(int point=0;point<(*ptFTemplates)[iTNum].iNumResultPoints;point++)
		{
			double x=(*ptFTemplates)[iTNum].ptFPoint[point].dLoc[0];
			double y=(*ptFTemplates)[iTNum].ptFPoint[point].dLoc[1];
			double correlation=(*ptFTemplates)[iTNum].ptFPoint[point].dScore;

			printf("       %s Match: (%7.3f, %7.3f)\n", point?"   A":"Best",x, y);
			printf("            Score: %6.4f\n", correlation);
			printf("            Angle: %7.3f\n",(*ptFTemplates)[iTNum].ptFPoint[point].dOrientation);
			printf("            Scale: %7.3f  %7.3f\n",	(*ptFTemplates)[iTNum].ptFPoint[point].dScale[0],
				(*ptFTemplates)[iTNum].ptFPoint[point].dScale[1]);
		}
	}
#endif

	return 0;
}


static void get_differentiators(
	vs_fid_data *diff_list,
	int size,
	int fid,
	double twidth,
	double theight,
	double /*hwidth*/,
	double theta,
	int dark_to_light)
{
	const int whole_list_size=6;
	const vs_fid_data whole_list[whole_list_size]=
	{
		{CIRCLE_FIDUCIAL,1,1,0,0,0},
		{CIRCLE_H_FIDUCIAL,1,1,.25,0,0},
		{SQUARE_FIDUCIAL,.8,.8,0,0,0},
		{PLUS_FIDUCIAL,1,1,.3,0,0},
		{TRIANGLE_UP_FIDUCIAL,1,1,0,0,0},
		{TRIANGLE_DOWN_FIDUCIAL,1,1,0,0,0}
	};

	int insert=0;

	for(int i=0;i<whole_list_size;i++)
	{
		if(whole_list[i].fid==fid && fid!=PLUS_FIDUCIAL)
			continue;

		// Because of angle, avoid both triangle type 
		if((whole_list[i].fid==TRIANGLE_UP_FIDUCIAL && fid==TRIANGLE_DOWN_FIDUCIAL)
			|| (whole_list[i].fid==TRIANGLE_DOWN_FIDUCIAL && fid==TRIANGLE_UP_FIDUCIAL))
			continue;

		if(!size)
			break;

		size--;

		if(whole_list[i].fid==PLUS_FIDUCIAL && fid==PLUS_FIDUCIAL)
		{
			diff_list[insert].fid=PLUS_FIDUCIAL_ROUNDED;
			diff_list[insert].twidth=1*twidth;
			diff_list[insert].theight=1*theight;
			diff_list[insert].hwidth=0.3*twidth;
			diff_list[insert].theta=0;
			diff_list[insert].dark_to_light=dark_to_light;
		}
		else
		{
			diff_list[insert].fid=whole_list[i].fid;
			diff_list[insert].twidth=whole_list[i].twidth*twidth;
			diff_list[insert].theight=whole_list[i].theight*theight;
			diff_list[insert].hwidth=whole_list[i].hwidth*twidth;
			diff_list[insert].theta=theta;
			diff_list[insert].dark_to_light=dark_to_light;
		}

		insert++;
	}

}
#ifdef __DEBUG_TEMPLATE
#ifdef WIN32
void write_JPEG_file (char * filename, int quality, BYTE *image_buffer, int image_height, int image_width)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE * outfile;		/* target file */
  JSAMPROW row_pointer[1];	/* pointer to JSAMPLE row[s] */
  int row_stride;		/* physical row width in image buffer */
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  if ((outfile = fopen(filename, "wb")) == NULL)
  {
    //fprintf(stderr, "can't open %s\n", filename);
	return;
  }
  jpeg_stdio_dest(&cinfo, outfile);

  cinfo.image_width = image_width; 	/* image width and height, in pixels */
  cinfo.image_height = image_height;
  cinfo.input_components = 1;		/* # of color components per pixel */
  cinfo.in_color_space = JCS_GRAYSCALE; 	/* colorspace of input image */
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

  jpeg_start_compress(&cinfo, TRUE);
  row_stride = image_width * 1;	/* JSAMPLEs per row in image_buffer */

  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = & image_buffer[cinfo.next_scanline * row_stride];
    (void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(outfile);

  jpeg_destroy_compress(&cinfo);
}
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Modified version for for 2D-SPI application
//////////////////////////////////////////////////////////////////////////////

// This function will initialize a vsfinder template and finder structure
const char *concreteVsWrapper::CreateVsFinderTemplate(	
	VsStFTemplate* ptFTemplate,		// Output: trained template
	VsStFinder* ptFinder,			// Output: initialized Finder structure
	templatetype tpl,				// Type of template to generate
	vs_fid_data *fid_data,			// array of fids to draw
	int num_fid_data,				// number of elements in the fid array
	vs_fid_poly_data *poly_data,	// polygon to draw
	double *min_scale,				// min scale array (x,y) to search for (0-?) =< max_scale, 1 is unity
	double *max_scale,				// max scale array (x,y) to search for (0-?) >= min_scale, 1 is unity
	double low_accept,				// Minimum score to accept in the Low Resolution image
	double high_accept,				// Minimum score to accept in the High Resolution image
	int iDepth,						// Depth of the template
	vs_fid_data *diff_list,			// List of difference patterns to distinguish 
	int diff_size,					// Size of difference patterns
	double mask_region)				// creates masked region that vsfind ignores. No effect if <=0

{
	// Decide the size of training image and create it
	int im_width = 0;
	int im_height = 0;

	if(diff_size>0 && diff_size<=8)
	{	// reduce the training image size to speed up the training
		// assume all diffential fiducail has the similar size
		im_width = (int)(diff_list[0].twidth * 4) + 20;
		im_height = (int)(diff_list[0].theight * 4) +20;
	}
	else
	{
		for(int i = 0; i < diff_size; i++)
		{
			im_width += (int)(diff_list[i].twidth * 2);
			im_height += (int)(diff_list[i].theight * 2);
		}
	}

	int width = 1;
	int height = 1;
	while(im_width >= width) width <<= 1;
	while(im_height >= height) height <<= 1;
	if(tpl == SKIPMARK || width == 1 || height == 1)
	{
		width = (int)m_width_pixels;
		height = (int)m_height_pixels;
	}

	VsEnviron oVisEnv = vswrapper::getEnv();

	VsCamImage cam_image=vsCreateCamImage( oVisEnv, 0, 0, (int)width, (int)height,
		VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, 1);

	// Create vs finder template
	VsStFTemplate   *ptFTemplates[]={ptFTemplate};
	VsStToolRect tool;
	const double dMinAngle=0;
	const double dMaxAngle=0;

	tool.dCenter[0]  = (width)/2.0;
	tool.dCenter[1]  = (height)/2.0;
	tool.dAngle	  = 0.0;

	fill_color fill_colors[2]={{65 , 192, 192}, {192 , 65, 65}};

	if(tpl == SKIPMARK)
	{
		fill_colors[0].fore_ground=128;
		fill_colors[0].back_ground=192;
		fill_colors[0].hollow=64;

		fill_colors[1].fore_ground=128;
		fill_colors[1].back_ground=64;
		fill_colors[1].hollow=192;
	}

	const char *ret=0;
	int fore_ground=0, back_ground=0, hollow=0;

	double start=clock();

	double twidth = 0.0;
	double theight = 0.0;

	for(int i=0;i<num_fid_data&&!ecfail(ret);i++)
	{
		fore_ground=fill_colors[!!fid_data[i].dark_to_light].fore_ground;
		back_ground=fill_colors[!!fid_data[i].dark_to_light].back_ground;
		hollow=fill_colors[!!fid_data[i].dark_to_light].hollow;

		twidth=max(twidth, fid_data[i].twidth);
		theight=max(theight, fid_data[i].theight);

		if(vsDrawFiducial(cam_image, fid_data[i].fid, tool.dCenter,
			fid_data[i].twidth, fid_data[i].theight, fid_data[i].hwidth, fore_ground,
				back_ground, hollow, !i, i)==-1)
		{
			if(tpl == SKIPMARK)
				ret= "666 Failed to draw skipmark template";
			else
				ret= "666 Failed to draw fiducial template";
		}
	}
	start=print_delta(1, "draw", start);

	if(!num_fid_data)
	{
		twidth=poly_data->width;
		theight=poly_data->height;

		fore_ground=fill_colors[!!poly_data->dark_to_light].fore_ground;
		back_ground=fill_colors[!!poly_data->dark_to_light].back_ground;

		int iCount = poly_data->poly.points().count();
		if(iCount!=3 && iCount!=6)
			ret = "Not a triangle or triangle frame"; 

		vector3 p0 = poly_data->poly.points().shift();
		vector3 p1 = poly_data->poly.points().shift();
		vector3 p2 = poly_data->poly.points().shift();

		double points[4][2] = {{p0.x,p0.y}, {p2.x,p2.y}, {p1.x,p1.y}, {p0.x,p0.y}};
		for(int i=0; i<4; i++)
		{
			points[i][0] = points[i][0] + width/2.0;
			points[i][1] = points[i][1] + height/2.0;
		}
		// Draw triangle fiducial/outside of triangle frame
		if(vsDrawConvexPolygon(cam_image, points, 3, fore_ground, back_ground, TRUE, FALSE) == -1)
		{
			if(tpl == SKIPMARK)
				ret = "666 Failed to draw Polygon skipmark template";
			else
				ret = "666 Failed to draw Polygon fiducial template";
		}

		// For triangle frame
		if(iCount == 6)
		{	// Draw inside triangle
			vector3 p3 = poly_data->poly.points().shift();
			vector3 p4 = poly_data->poly.points().shift();
			vector3 p5 = poly_data->poly.points().shift();
			double points2[4][2] = {{p3.x,p3.y}, {p5.x,p5.y}, {p4.x,p4.y}, {p3.x,p3.y}};
			for(int i=0; i<4; i++)
			{
				points2[i][0] = points2[i][0] + width/2.0;
				points2[i][1] = points2[i][1] + height/2.0;
			}
			VsCamImage temp_image=vsCreateCamImage( oVisEnv, 0, 0, (int)width, (int)height,
				VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, 1);

			if(vsDrawConvexPolygon(temp_image, points2, 3, fore_ground, back_ground, TRUE, FALSE) == -1)
			{
				if(tpl == SKIPMARK)
					ret = "666 Failed to draw Polygon skipmark template";
				else
					ret = "666 Failed to draw Polygon fiducial template";
			}

			// Create triangle frame (background need to be be compensated 
			ImageClipSub(cam_image, temp_image, back_ground);

			vsDispose(temp_image);
		}
	}

	if(tpl == SKIPMARK)
	{
		// Skipmarks require much less background than fiducials
		// this is because we are less concerned with edges
		tool.dWidth	= twidth  + 2*SKIPMARK_BACKGROUND;
		tool.dHeight = theight + 2*SKIPMARK_BACKGROUND;
	}
	else
	{
		tool.dWidth	= twidth  * 1.4; // add a little background to the template.
		tool.dHeight = theight * 1.4; // This is a must.
	}

	if(ecfail(ret))
	{
	}
	else if (vsCreateFTemplate (cam_image, &tool, iDepth,
		dMinAngle, dMaxAngle, min_scale, max_scale, ptFTemplate) == -1)
			ret=lookup_template_error(ptFTemplate->iResultFlags); // fail
	else if(ecfail(ret=create_keepout(oVisEnv, *ptFTemplate, fid_data, num_fid_data, mask_region)))
	{
	}
	else
	{
		start=print_delta(1, "create template", start);
		/* Set some of the template properties - even though most of them
		are the default values already set by vsCreateFTemplate() */
		ptFTemplate->iPyramidType = VS_FINDER_PYRAMID_AVERAGE;
		ptFTemplate->yUniformScaling = TRUE;
		ptFTemplate->iSpeed = 90; //-1; /* auto set */
		ptFTemplate->iTemplateType = VS_FINDER_GRAY_SCALE_BASED;
		ptFTemplate->yMultiLayerBuildup = FALSE;
		ptFTemplate->iAccuracy = 50;
		ptFTemplate->iIgnoreValuesAbove = 255; /* default */
		ptFTemplate->iIgnoreValuesBelow = 0;
		ptFTemplate->iCorrelationType = 1; /* gain and offset */
		ptFTemplate->dGainFactor = 40.0;
		ptFTemplate->iOffsetValue = 255; /* ignored if gain only */
		ptFTemplate->yAllowNegatives = TRUE; /* enable inverse match*/
		ptFTemplate->yAllowPyramidTypeChange = 0;
		ptFTemplate->yAllowCorrelationTypeChange = FALSE ;
		ptFTemplate->iMinimumPyramidDepth = 1;
		ptFTemplate->dLoResMinScore     = 0.50;
		ptFTemplate->dHiResMinScore     = 0.50;
		ptFTemplate->iMaxResultPoints   = 2; /* # matches to find */
		//ptFTemplate->yComputeTrueNgcScore = TRUE;

		// Setting the training space containing other fiducials or objects of interest that the finder
		tool.dWidth = width;
		tool.dHeight = height;
		tool.dAngle	= 0;

		int half_diff=(int)((diff_size/2.0)+.5);

		double diff_center[2]={5, 0};
		int the_one=0;
		for(int diff=0;diff<diff_size;diff++)
		{
			// Differential fiducial Y location
			if(diff<half_diff)
				diff_center[1] = 5 + diff_list[the_one].theight/2;
			else
				diff_center[1] = height-1-5-diff_list[the_one].theight/2;

			if(diff==half_diff)
			{
				diff_center[0]=5;
			}

			if(diff_center[1]-diff_list[the_one].theight/2<=0)
				continue;
			if(diff_center[1]+diff_list[the_one].theight/2>=height)
				continue;

			diff_center[0]+=diff_list[the_one].twidth/2;

			if(fiducial_intersect(diff_center, diff_list[the_one].twidth, diff_list[the_one].theight,
				tool.dCenter, twidth, theight))
					diff_center[0]=tool.dCenter[0]+twidth+diff_list[the_one].twidth/2;

			if(diff_center[0]+diff_list[the_one].twidth/2.>=width)
				continue;

			if(vsDrawFiducial(cam_image, diff_list[the_one].fid, diff_center,
				diff_list[the_one].twidth, diff_list[diff].theight, diff_list[the_one].hwidth, fore_ground,
					back_ground, back_ground, 0, 1)==-1)
						ret= "666 Failed to draw difference shape";

			diff_center[0]+=diff_list[the_one].twidth/2+30;
			the_one++;
		}
		/*
		static int i=5;
		VsStFileIOControl tFileControl;
		tFileControl.eFileType = VS_FFORMAT_GIF;
		char cImageName[255];
		sprintf(cImageName,"%s_%d.gif", "C:\\Temp\\fiducialImag.gif", i++);
		tFileControl.pcFileName = cImageName;
		vsSaveImageData(cam_image, &tFileControl);
		//*/

#ifdef __DEBUG_TEMPLATE
// used for debug to save the image out to a file
// ajrajr
		VsStCamImageInfo tData;
		vsInqCamImageInfo(cam_image, &tData);
#ifdef WIN32
		write_JPEG_file ("afid.jpg", 100, tData.pbBuf, height, width);
#else
		write_image(tData.pbBuf, width, height, 0, 0, "afid.ccf", 0, 0, 0, 1, 1, "help me");
#endif // WIN32

#endif

		if (vsCreateFinder(cam_image, &tool, 1, ptFTemplates, ptFinder) == -1)
			ret= lookup_finder_error(ptFinder->iResultFlags);
		else
		{
			ptFTemplate->dLoResMinScore     = low_accept;//0.50;
			ptFTemplate->dHiResMinScore     = high_accept;//0.60;
		}
	}

	vsDispose(cam_image);

	vswrapper::releaseEnv();

	return ret;
}

const char *concreteVsWrapper::CreateVsFinderTemplate(
	VsStFTemplate* ptFTemplate,		// Output: trained template
	VsStFinder* ptFinder,			// Output: initialized Finder structure
	templatetype tpl, int fid, double twidth,
	double theight, double hwidth, double theta, int dark_to_light,
	double *min_scale, double *max_scale, double low_accept,
	double high_accept,	int depth, double mask_region)
{
	vs_fid_data fid_data;
	const int diff_size=5;
	vs_fid_data diff_list[diff_size];

	fid_data.fid=fid;
	fid_data.twidth=twidth;
	fid_data.theight=theight;
	fid_data.hwidth=hwidth;
	fid_data.theta=theta;
	fid_data.dark_to_light=dark_to_light;

	if(fabs(theta) > 0.0) 
	{
		if(fabs(fabs(theta)*4-2)>0.5) // Make sure theta=+-0.5 (180 degree) is excluded 
		{
			fid_data.twidth = theight;
			fid_data.theight = twidth;
		}
	}

	if(tpl == FIDUCIAL)
		get_differentiators(diff_list, diff_size, fid, twidth, theight,
			hwidth, theta, dark_to_light);

	return CreateVsFinderTemplate(ptFTemplate, ptFinder,
		tpl, &fid_data, 1, 0, min_scale, max_scale, low_accept,
		high_accept, depth, diff_list, diff_size, mask_region);
}


const char * concreteVsWrapper::create_disc_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl, double r,
	double theta, int dark_to_light, 
	double *min_scale, double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{	
 	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0)
	{
		r/=m_pixel_size;
		double twidth=r*2, theight=r*2;

		if(tpl == SKIPMARK)
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, CIRCLE_H_FIDUCIAL, twidth+2*SKIPMARK_EDGE,
				theight+2*SKIPMARK_EDGE, SKIPMARK_EDGE, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
		else
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, CIRCLE_FIDUCIAL, twidth, theight, 0, theta,
				dark_to_light, min_scale, max_scale, low_accept, high_accept, depth, mask_region);
	}
	else
	{
		return "600 bad pixel size";
	}
}


const char * concreteVsWrapper::create_rectangle_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl,
	double base, double height, double theta, int dark_to_light,
	double *min_scale, double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		base/=m_pixel_size;
		height/=m_pixel_size;

		if(tpl == SKIPMARK)
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, SQUARE_H_FIDUCIAL, base+2*SKIPMARK_EDGE,
				height+2*SKIPMARK_EDGE, SKIPMARK_EDGE, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
		else
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, SQUARE_FIDUCIAL, base,
				height, 0, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_rectangleframe_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl,
	double base, double height, double thickness, double theta, int dark_to_light,
	double *min_scale, double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		base/=m_pixel_size;
		height/=m_pixel_size;
		thickness /= m_pixel_size;

		return CreateVsFinderTemplate(ptFTemplate, ptFinder,
			tpl, SQUARE_H_FIDUCIAL, base,
			height, thickness, theta, dark_to_light, min_scale, max_scale,
			low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_diamond_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl,
	double base, double height, double theta, int dark_to_light,
	double *min_scale, double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		base/=m_pixel_size;
		height/=m_pixel_size;

		if(tpl == SKIPMARK)
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, DIAMOND_H_FIDUCIAL, base+2*SKIPMARK_EDGE,
				height+2*SKIPMARK_EDGE, SKIPMARK_EDGE, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
		else
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, DIAMOND_FIDUCIAL, base,
				height, 0, theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_diamondframe_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl,
	double base, double height, double thickness, double theta, int dark_to_light,
	double *min_scale, double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		base/=m_pixel_size;
		height/=m_pixel_size;
		thickness/=m_pixel_size;

		return CreateVsFinderTemplate(ptFTemplate, ptFinder,
			tpl, DIAMOND_H_FIDUCIAL, base,
			height, thickness, theta, dark_to_light, min_scale, max_scale,
			low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_triangle_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl,
	double base, double height, double offset, double theta, int dark_to_light, 
	double *min_scale,  double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		vs_fid_poly_data poly_data;

		// Convert from meters to pixels.
		base/=m_pixel_size;
		height/=m_pixel_size;
		offset/=m_pixel_size; // offset to x of top point of triangle

		// Create rotation matrix from theta
		matrix3 rotation = matrix3().rotate(2, (2 * M_PI * -theta));

		polygon triangle;

		//Center of triange bounding box.
		vector3 bound_center = rotation * vector3(base/2.0,height/2.0);

		triangle.points().push(vector3(0,0));
		triangle.points().push(vector3(base, 0));
		triangle.points().push(vector3(offset, height));
		triangle *= rotation;
		rect tri_bound = triangle.bound();
		base = tri_bound.width();
		height = tri_bound.height();

		vector3 tri_cent = triangle.centroid();
		vector3 delta = bound_center - tri_cent;

		// move centroid of triangle to origin
		triangle-=triangle.centroid();

		// Now triangle centroid is centered in fov.
		// Need to ensure template is big enough.
		// Delta is difference between triangle bounding box and centrod.
		// So base would be enough if centered, but we need extra delta
		// on at least once side and both gives some border.
		poly_data.width = base + (2 * fabs(delta.x));
		poly_data.height = height + (2 * fabs(delta.y));
		poly_data.poly = triangle;
		poly_data.dark_to_light = dark_to_light;

		vs_fid_data fid_data;
		const int diff_size=5;
		vs_fid_data diff_list[diff_size];

		fid_data.fid=TRIANGLE_UP_H_FIDUCIAL;
		fid_data.twidth=base;
		fid_data.theight=height;
		fid_data.hwidth=0;
		fid_data.theta=theta;
		fid_data.dark_to_light=dark_to_light;

		if(tpl == FIDUCIAL)
			get_differentiators(diff_list, diff_size-1, TRIANGLE_UP_FIDUCIAL, base, height,
				0, theta, dark_to_light);

		return CreateVsFinderTemplate(ptFTemplate, ptFinder,
			tpl, 0, 0, &poly_data, min_scale, max_scale,
			low_accept, high_accept, depth, diff_list, diff_size-1, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char* concreteVsWrapper::create_triangleFrame_template1(
	int* piNodeID,			// Output: nodeID of map	
	templatetype tpl,
	double base, double height, double offset, double thickness, double theta, int dark_to_light, 
	double *min_scale,  double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		vs_fid_poly_data poly_data;

		// Convert from meters to pixels.
		base/=m_pixel_size;
		height/=m_pixel_size;
		offset/=m_pixel_size; // offset to x of top point of triangle
		thickness/=m_pixel_size;

		// Create rotation matrix from theta
		matrix3 rotation = matrix3().rotate(2, (2 * M_PI * -theta));

		polygon triangleFrame;

		//Center of triange bounding box.
		vector3 bound_center = rotation * vector3(base/2.0,height/2.0);

		// Outside triangle
		triangleFrame.points().push(vector3(0,0));
		triangleFrame.points().push(vector3(base, 0));
		triangleFrame.points().push(vector3(offset, height));
		// Inside triangle (for equilateral triangle)
		triangleFrame.points().push(vector3(sqrt(3.0)*thickness, thickness));
		triangleFrame.points().push(vector3(base-sqrt(3.0)*thickness, thickness));
		triangleFrame.points().push(vector3(offset, height-2*thickness));

		triangleFrame *= rotation;
		rect tri_bound = triangleFrame.bound();
		base = tri_bound.width();
		height = tri_bound.height();

		vector3 tri_cent = triangleFrame.centroid();
		vector3 delta = bound_center - tri_cent;

		// move centroid of triangleFrame to origin
		triangleFrame-=triangleFrame.centroid();

		// Now triangleFrame centroid is centered in fov.
		// Need to ensure template is big enough.
		// Delta is difference between triangleFrame bounding box and centrod.
		// So base would be enough if centered, but we need extra delta
		// on at least once side and both gives some border.
		poly_data.width = base + (2 * fabs(delta.x));
		poly_data.height = height + (2 * fabs(delta.y));
		poly_data.poly = triangleFrame;
		poly_data.dark_to_light = dark_to_light;

		vs_fid_data fid_data;
		const int diff_size=5;
		vs_fid_data diff_list[diff_size];

		fid_data.fid=TRIANGLE_UP_H_FIDUCIAL;
		fid_data.twidth=base;
		fid_data.theight=height;
		fid_data.hwidth=0;
		fid_data.theta=theta;
		fid_data.dark_to_light=dark_to_light;

		if(tpl == FIDUCIAL)
			get_differentiators(diff_list, diff_size-1, TRIANGLE_UP_FIDUCIAL, base, height,
				0, theta, dark_to_light);

		return CreateVsFinderTemplate(ptFTemplate, ptFinder,
			tpl, 0, 0, &poly_data, min_scale, max_scale,
			low_accept, high_accept, depth, diff_list, diff_size-1, mask_region);
	}
	return "Not implemented yet";
}

const char * concreteVsWrapper::create_donut_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl,
	double inner_radius, double outer_radius, double theta, int dark_to_light, 
	double *min_scale, double *max_scale, double low_accept, double high_accept, 
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		inner_radius/=m_pixel_size;
		outer_radius/=m_pixel_size;

		double twidth=outer_radius*2, theight=outer_radius*2;

		return CreateVsFinderTemplate(ptFTemplate, ptFinder,
			tpl, CIRCLE_H_FIDUCIAL, twidth, theight,
			outer_radius-inner_radius, theta, dark_to_light, min_scale, max_scale,
			low_accept, high_accept, depth, mask_region);
	}
	else
		return "600 bad pixel size";
}

const char * concreteVsWrapper::create_cross_template(
	int* piNodeID,			// Output: nodeID of map
	templatetype tpl,
	double base, double height, double base_leg, double height_leg, int rounded_edges, double theta, int dark_to_light, 
	double *min_scale, double *max_scale, double low_accept, double high_accept,
	double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		if(rounded_edges != 0)
		{
			base/=m_pixel_size;
			height/=m_pixel_size;

			if(fabs(base_leg-height_leg) > DBL_EPSILON)
				return "500 Rounded-edge Cross Fiducials with different width legs are not valid";

			base_leg/=m_pixel_size;

			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, PLUS_FIDUCIAL_ROUNDED, base, height, base_leg,
				theta, dark_to_light, min_scale, max_scale, low_accept, high_accept, depth,
				mask_region);
		}
		else
		{
			vs_fid_data fid_data[2];

			base/=m_pixel_size;
			height/=m_pixel_size;
			height_leg/=m_pixel_size;
			base_leg/=m_pixel_size;

			fid_data[0].fid=SQUARE_FIDUCIAL;
			fid_data[0].twidth=base;
			fid_data[0].theight=base_leg;
			fid_data[0].hwidth=0;
			fid_data[0].theta=theta;
			fid_data[0].dark_to_light=dark_to_light;

			fid_data[1].fid=SQUARE_FIDUCIAL;
			fid_data[1].twidth=height_leg;
			fid_data[1].theight=height;
			fid_data[1].hwidth=0;
			fid_data[1].theta=theta;
			fid_data[1].dark_to_light=dark_to_light;

			const int diff_size=5;
			vs_fid_data diff_list[diff_size];

			get_differentiators(diff_list, diff_size, PLUS_FIDUCIAL, base, height,
				0, theta, dark_to_light);

			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, fid_data, 2, 0, min_scale, max_scale,
				low_accept, high_accept, depth, diff_list, diff_size, mask_region);
		}
	}
	else
		return "600 bad pixel size";
}

const char* concreteVsWrapper::create_checkerpattern_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double theta, int dark_to_light, 
		double *min_scale, double *max_scale, double low_accept, double high_accept,
		double mask_region, int depth)
{
	vs_template_finder t;
	t.ptFTemplate = new VsStFTemplate;
	t.ptFinder = new VsStFinder;

	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		m_templateMap[m_iCurrentNodeID] = t;
		*piNodeID = m_iCurrentNodeID;
		m_iCurrentNodeID++;

		ReleaseMutex(m_hMutex);
	}

	VsStFTemplate* ptFTemplate = t.ptFTemplate;
	VsStFinder* ptFinder = t.ptFinder;

	assert( m_pixel_size > 0 );
	if( m_pixel_size > 0 )
	{
		base/=m_pixel_size;
		height/=m_pixel_size;

		if(theta == 0 || fabs(fabs(theta)-0.5)<0.125) // 0 or 180 degree
		{
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, CHECKER_UL_FIDUCIAL, base, height, 0,
				theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region);
		}
		else
		{
			// with angle 90, the upper-left will be upper right, 
			// the switch of base and height will be in the function
			return CreateVsFinderTemplate(ptFTemplate, ptFinder,
				tpl, CHECKER_UR_FIDUCIAL, base, height, 0,
				theta, dark_to_light, min_scale, max_scale,
				low_accept, high_accept, depth, mask_region); 
		}
	}
	else
		return "600 bad pixel size";
}

const char* concreteVsWrapper::Find(
	int iNodeID,			// Input: nodeID of map	
	unsigned char *image, int width, int height,
	double &x, double &y, double &correlation, double &ambig, double *ngc,
	double search_center_x, double search_center_y, double search_width,
	double search_height, double time_out, int y_origin_ll, double min_accept,
	double max_accept, int num_finds)
{
	double start=clock();

	VsStFTemplate* ptFTemplate;
	VsStFinder* ptFinder;
	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		ptFTemplate = m_templateMap[iNodeID].ptFTemplate;
		ptFinder = m_templateMap[iNodeID].ptFinder;
		
		ReleaseMutex(m_hMutex);
	}

	VsEnviron oVisEnv = vswrapper::getEnv();
	VsStFTemplate *ptFTemplates[]={ptFTemplate};
	VsCamImage cam_image=vsCreateCamImageFromBuffer( oVisEnv, 0, image, width, height,
		width, VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, y_origin_ll);

	if(ngc)
		ptFTemplate->yComputeTrueNgcScore = TRUE;

	const char *res=0;

	/* set the search space for vsFind() */
	ptFinder->tToolRect.dCenter[0]  =  search_center_x;
	ptFinder->tToolRect.dCenter[1]  = search_center_y;
	ptFinder->tToolRect.dWidth	  = search_width;
	ptFinder->tToolRect.dHeight	  = search_height;
	ptFinder->tToolRect.dAngle	  = 0.0;

	if(min_accept>0) ptFTemplate->dLoResMinScore=min_accept;
	if(max_accept>0) ptFTemplate->dHiResMinScore=max_accept;
	if(num_finds>0)  ptFTemplate->iMaxResultPoints=num_finds;

	res=find_template(ptFTemplates, 1, cam_image, ptFinder, time_out);
	
	start=print_delta(1, "found", start);

	if(ptFTemplate->ptFPoint && ptFTemplate->ptFPoint[0].dScore!=-1)
	{
		x=ptFTemplate->ptFPoint[0].dLoc[0];
		y=ptFTemplate->ptFPoint[0].dLoc[1];
		correlation=ptFTemplate->ptFPoint[0].dScore;
		if(ptFTemplate->iNumResultPoints>1)
			ambig=ptFTemplate->ptFPoint[1].dScore/correlation;
		else
			ambig=0;

		if(ngc)
			*ngc=ptFTemplate->ptFPoint[0].dNgcScore;
	}
	else
	{
		y=x=0;
		correlation=0;
		ambig=1;
		if(ngc) *ngc=0;
	}
	
	vsDispose(cam_image);

	print_delta(1, "vsDisposes", start);

	vswrapper::releaseEnv();

	return res;
}

void concreteVsWrapper::ClearTemplateMap()
{
	map<int, vs_template_finder>::iterator it;
	
	if (WaitForSingleObject(m_hMutex,INFINITE)==WAIT_OBJECT_0)
	{
		for( it = m_templateMap.begin(); it != m_templateMap.end(); it++)
		{
			vsDispose(it->second.ptFTemplate);
			vsDispose(it->second.ptFinder);

			delete it->second.ptFTemplate;
			delete it->second.ptFinder;
		}
		m_templateMap.clear();

		m_iCurrentNodeID = 0;
		ReleaseMutex(m_hMutex);
	}
}

