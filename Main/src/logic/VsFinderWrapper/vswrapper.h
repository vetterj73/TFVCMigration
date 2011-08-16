// 
// File: vswrapper.h
// Description: Interface file for vswrapper.lib
// Copyright (c) 2002 CyberOptics Corp. 
//
// $Author: Dbutler $ 
// $Revision: 9 $
// $Header: /Ghidorah/QNX6/Realtime/vswrapper.h 9     9/13/05 3:54p Dbutler $

#ifndef __VSWRAPPER__
#define __VSWRAPPER__
// wrapper functions to generate fiducial templates for the SE 300. The parameters and 
// shapes are defined in the same sense as the SMEMA shape definitions.
//
// Unless otherwise stated the units are assumed to standard SI units.

// All Return values a character strings that one can interpret using errcode.h


// This module is ment for use with the SE 300. Pixels are assumed to be square and
// 20 um.

#include <windows.h>

#include "shape.h"

class concreteVsWrapper;
typedef void *VsEnviron;

struct vs_fid_data
{
	int fid;       // Enum or define for vs_draw_fiducial function
	double twidth;  // width of the fiducial
	double theight;  // height of the fiducial
	double hwidth;  // width of the "ring" if the type is hollow.
	double theta;   // rotation, not currently used
	int dark_to_light;
};

struct vs_fid_poly_data
{
	polygon poly;		// The polygon 
	double width;		// Width of the fiducial in pixels
	double height;		// Height of the fiducial in pixels
	int dark_to_light;
};

class vswrapper
{
public:

	vswrapper( concreteVsWrapper* concrete=0 );
	virtual ~vswrapper();

	enum templatetype
	{
		FIDUCIAL,
		SKIPMARK
	};

	// Pixel & FOV dimensions should be set before any other functions are called
	void set_pixel_size(double size);
	void set_fov_cols(int cols);
	void set_fov_rows(int rows);

	static VsEnviron getEnv(); // Get environment for calling thread.
	static void releaseEnv();	// Match every call to get() with one to release().
	static void disposeEnv();

private:
	concreteVsWrapper * concreteWrapper_;

	static VsEnviron& getStaticEnv(); // Get environment for calling thread.
	static DWORD&     getEnvThread(); // Get thread ID of the owner thread
	static int        getAddEnvUseCount(int add=0); // Get the use count for this thread and then add the supplied value

///////////////////////////////////////////////////////////////////////////////
// Modified version for for 2D-SPI application
//////////////////////////////////////////////////////////////////////////////
public:
	// create a template for locating a disc
	virtual const char * create_disc_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double r,				// radius of the disc in meters
		double theta,			// rotation about the center of the shape (Useless and can be any value)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a donut
	virtual const char * create_donut_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double inner_radius,
		double outer_radius,
		double theta,			// rotation about the center of the shape (Useless and can be any value)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a cross
	virtual const char * create_cross_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double base_leg,
		double height_leg,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		int rounded_edges=0,	// If !0, the edges of the cross will be rounded
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a diamond
	virtual const char * create_diamond_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a diamond frame
	virtual const char * create_diamondframe_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double thick,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a triangle
	virtual const char * create_triangle_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double offset,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a triangle frame (equilateral)
	virtual const char * create_triangleframe_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double offset,
		double thick,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a rectangle
	virtual const char * create_rectangle_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a rectangle frame
	virtual const char * create_rectangleframe_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double thick,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a checkerpattern
	virtual const char * create_checkerpattern_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double theta,			// rotation about the center of the shape (only for 90*n degee, 360degree = 1 unit, clockwise)
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// load the template by a given name and find the match in the given image.
	virtual const char* Find(
		int iNodeID,			// map ID of template  and finder		
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
		int y_origin_ll=1,      // If !0 origin in start of the image is 0,0 increasing in Y and X as you move forward
		double min_accept=-1,   // If >0 minimum score to persue at min pyramid level to look for peak override
		double max_accept=-1,   // If >0 minumum score to accept at max pyramid level to look for peak override
		int num_finds=-1        // If > 0 number of versions of the template to find
	);

};

#endif
