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

	// Pixel & FOV dimensions should be set before any other functions are called
	void set_pixel_size(double size) { m_pixel_size = size; }
	void set_fov_cols(int cols) { m_width_pixels = cols; }
	void set_fov_rows(int rows) { m_height_pixels = rows; }

	double m_pixel_size, m_width_pixels, m_height_pixels;
	
	VsStFTemplate * pFTemplate_;

///////////////////////////////////////////////////////////////////////////////
// Modified version for for 2D-SPI application
//////////////////////////////////////////////////////////////////////////////

public:
	const char* create_disc_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl, 
		double r, double theta, 
		int dark_to_light, bool bAllowNegativeMatch, 
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);	// Output: trained template and initialized Finder structure)

	const char* create_rectangle_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double theta, 
		int dark_to_light, bool bAllowNegativeMatch,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_rectangleframe_template(
		int* piNodeID,			// Output: nodeID of map
		templatetype tpl,
		double base, double height, double thickness, double theta, 
		int dark_to_light, bool bAllowNegativeMatch,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_diamond_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double theta, 
		int dark_to_light, bool bAllowNegativeMatch,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_diamondframe_template(
		int* piNodeID,			// Output: nodeID of map
		templatetype tpl,
		double base, double height, double thickness, double theta, 
		int dark_to_light, bool bAllowNegativeMatch,
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_triangle_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double offset, double theta, 
		int dark_to_light, bool bAllowNegativeMatch, 
		double *min_scale,  double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_triangleFrame_template1(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double offset, double thickness, double theta, 
		int dark_to_light, bool bAllowNegativeMatch, 
		double *min_scale,  double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_donut_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double inner_radius, double outer_radius, double theta, 
		int dark_to_light, bool bAllowNegativeMatch, 
		double *min_scale, double *max_scale, double low_accept, double high_accept, 
		double mask_region, int depth);

	const char* create_cross_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double base_leg, double height_leg, int rounded_edges, 
		double theta, int dark_to_light, bool bAllowNegativeMatch, 
		double *min_scale, double *max_scale, double low_accept, double high_accept,
		double mask_region, int depth);

	const char* create_checkerpattern_template(
		int* piNodeID,			// Output: nodeID of map	
		templatetype tpl,
		double base, double height, double theta, 
		int dark_to_light, bool bAllowNegativeMatch, 
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
		double theight, double hwidth, double theta, 
		int dark_to_light, bool bAllowNegativeMatch,
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
