#ifndef __CONCRETE_VSWRAPPER__
#define __CONCRETE_VSWRAPPER__

#include "vswrapper.h"
#include "vision.h"
#include <map>
using std::map;


struct vs_template_finder
{
	VsStFTemplate* ptFTemplate;
	VsStFinder* ptFinder;
};

class concreteVsWrapper : public vswrapper
{
public:

	concreteVsWrapper();
	~concreteVsWrapper();

	//enum templatetype
	//{
	//	FIDUCIAL,
	//	SKIPMARK
	//};

	// Pixel & FOV dimensions should be set before any other functions are called
	void set_pixel_size(double size) { m_pixel_size = size; }
	void set_fov_cols(int cols) { m_width_pixels = cols; }
	void set_fov_rows(int rows) { m_height_pixels = rows; }
/*
	// create a template for locating a disc
	const char * create_disc_template(
		const char *file_name,	// name of the template file to save
		templatetype tpl,		// Type of template to generate
		double r,				// radius of the disc in meters
		double theta,			// (Not currently supported) rotation about the center of the shape
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a cross
	const char * create_cross_template(
		const char *file_name,	// name of the template file to save
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double base_leg,
		double height_leg,
		double theta,			// (Not currently supported) rotation about the center of the shape
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		int rounded_edges=0,	// If !0, the edges of the cross will be rounded
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a cross
	const char * create_diamond_template(
		const char *file_name,	// name of the template file to save
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double theta,			// (Not currently supported) rotation about the center of the shape
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// creates a template for locating a cross
	const char * create_donut_template(
		const char *file_name,	// name of the template file to save
		templatetype tpl,		// Type of template to generate
		double inner_radius,
		double outer_radius,
		double theta,			// (Not currently supported) rotation about the center of the shape
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	const char * create_triangle_template(
		const char *file_name,	// name of the template file to save
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double offset,
		double theta,			// (Not currently supported) rotation about the center of the shape
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	const char * create_rectangle_template(
		const char *file_name,	// name of the template file to save
		templatetype tpl,		// Type of template to generate
		double base,
		double height,
		double theta,			// (Not currently supported) rotation about the center of the shape
		int dark_to_light,		// If not !0 assumes dark back ground and light fiducial
		double *min_scale,		// An array with two elements (x,y) for min scale 0-1
		double *max_scale,		// An array with two elements (x,y) for max scale 0-1 max_scale[]>=min_scale[]
		double low_accept=0.5,	// Minimum score to accept in the Low Resolution image
		double high_accept=0.6,	// Minimum score to accept in the High Resolution image
		double mask_region=0,	// creates a masked region that vsFind will ignore. No effect if <=0
		int depth=3				// number of pyramid level to use
	);

	// load the template by a given name and find the match in the given image.
	const char * load_and_find(
		const char *file_name,        // name of the template file
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

	// loads template from file
	int load_finder_template(
		const char *file_name
	);

	// performs vsfind given template object.
	const char * find_from_template(
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
*/
private:
/*
	// This function will only draw one fiducial per template
	const char *create_and_save(
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
		vs_fid_data *diff_list=0,
		int diff_size=0,
		double mask_region=0			// creates masked region that vsfind ignores. No effect if <=0
		);

	const char *create_and_save(
		const char *template_name, 
		templatetype tpl, 
		int fid, 
		double twidth, 
		double theight, 
		double hwidth, 
		double theta, 
		int dark_to_light, 
		double *min_scale, 
		double *max_scale, 
		double low_accept, 
		double high_accept,	
		int depth, 
		double mask_region=0
		);*/

	double m_pixel_size, m_width_pixels, m_height_pixels;
	
	VsStFTemplate * pFTemplate_;

///////////////////////////////////////////////////////////////////////////////
// Modified version for for 2D-SPI application
//////////////////////////////////////////////////////////////////////////////

public:
	const char* create_disc_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl, double r,
		double theta, int dark_to_light, 
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);	// Output: trained template and initialized Finder structure)

	const char* create_rectangle_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double theta, int dark_to_light,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_rectangleframe_template(
		int* piNodeID,			// Output: nodeID of map
		templatetype tpl,
		double base, double height, double thick, double theta, int dark_to_light,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_diamond_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double theta, int dark_to_light,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_diamondframe_template(
		int* piNodeID,			// Output: nodeID of map
		templatetype tpl,
		double base, double height, double thick, double theta, int dark_to_light,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_triangle_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double offset, double theta, int dark_to_light, 
		double *min_scale,  double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_triangleFrame_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double offset, double thick, double theta, int dark_to_light, 
		double *min_scale,  double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_donut_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double inner_radius, double outer_radius, double theta, int dark_to_light, 
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_cross_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double base_leg, double height_leg, int rounded_edges, double theta, int dark_to_light, 
		double *min_scale, double *max_scale, double low_accept, double high_accept,
		double mask_region, int depth);

	const char* create_checkerpattern_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double theta, int dark_to_light, 
		double *min_scale, double *max_scale, double low_accept, double high_accept,
		double mask_region, int depth);

	const char* Find(
		int iNodeID,			// Input: nodeID of map	
		unsigned char *image, int width, int height,
		double &x, double &y, double &correlation, double &ambig, double *ngc,
		double search_center_x, double search_center_y, double search_width,
		double search_height, double time_out, int y_origin_ll, double min_accept,
		double max_accept, int num_finds);

private:
	// This function will initialize a vsfinder template and finder structure
	const char* CreateVsFinderTemplate(
		VsStFTemplate* ptFTemplates,	// Output: trained template
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
		double mask_region);				// creates masked region that vsfind ignores. No effect if <=0

	const char* CreateVsFinderTemplate(
		VsStFTemplate* ptFTemplate,		// Output: trained template
		VsStFinder* ptFinder,			// Output: initialized Finder structure)
		templatetype tpl, int fid, double twidth,
		double theight, double hwidth, double theta, int dark_to_light,
		double *min_scale, double *max_scale, double low_accept,
		double high_accept,	int depth, double mask_region);

public:
	void ClearTemplateMap();

private:
	// thread safety
	HANDLE m_hMutex;
	map<int, vs_template_finder> m_templateMap;
	int m_iCurrentNodeID;
};


#endif // __CONCRETE_VSWRAPPER__
