#include "vswrapper.h"
#include "concreteVswrapper.h"
#include <assert.h>

vswrapper::vswrapper( concreteVsWrapper* concrete ) : 
	concreteWrapper_(concrete)
{
	if( !concrete )
		concreteWrapper_ = new concreteVsWrapper();
}

vswrapper::~vswrapper() 
{ 
	if(concreteWrapper_ && concreteWrapper_ != this) 
	{
		delete concreteWrapper_; 
	}
}


void vswrapper::set_pixel_size(double size) 
{ 
	if( concreteWrapper_ ) 
		concreteWrapper_->set_pixel_size( size );
}

void vswrapper::set_fov_cols(int cols)
{ 
	if( concreteWrapper_ ) 
		concreteWrapper_->set_fov_cols( cols );
}

void vswrapper::set_fov_rows(int rows)
{ 
	if( concreteWrapper_ ) 
		concreteWrapper_->set_fov_rows( rows );
}
/*
const char * vswrapper::create_disc_template(
	const char *file_name, templatetype tpl, double r, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_disc_template(
			file_name, tpl, r, theta, 
			dark_to_light, min_scale, max_scale, 
			low_accept, high_accept, mask_region, depth );
	return retVal;
}

// creates a template for locating a cross
const char * vswrapper::create_cross_template(
	const char *file_name, templatetype tpl, double base, double height, 
	double base_leg, double height_leg, double theta, int dark_to_light,
	double *min_scale, double *max_scale, int rounded_edges, 
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_cross_template(
			file_name, tpl, base, height,
			base_leg, height_leg, theta, dark_to_light,
			min_scale, max_scale, rounded_edges,
			low_accept, high_accept, mask_region, depth );
	return retVal;
}


// creates a template for locating a cross
const char * vswrapper::create_diamond_template(
	const char *file_name, templatetype tpl, double base, double height, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_diamond_template(
			file_name, tpl, base, height, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );
	return retVal;
}

// creates a template for locating a cross
const char * vswrapper::create_donut_template(
	const char *file_name, templatetype tpl, double inner_radius, double outer_radius, double theta, 
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_donut_template(
			file_name, tpl, inner_radius, outer_radius, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );
	return retVal;
}


const char * vswrapper::create_triangle_template(
	const char *file_name, templatetype tpl, double base, double height, double offset, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_triangle_template(
			file_name, tpl, base, height, offset, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );
	return retVal;
}


const char * vswrapper::create_rectangle_template(
	const char *file_name, templatetype tpl, double base, double height, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_rectangle_template(
			file_name, tpl, base, height, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );
	return retVal;
}

// load the template by a given name and find the match in the given image.
const char * vswrapper::load_and_find(
	const char *file_name, unsigned char *image, int width, int height,
	double &x, double &y, double &correlation, double &ambig, double *ngc,
	double search_center_x, double search_center_y, double search_width, double search_height,
	double time_out, int y_origin_ll, double min_accept, double max_accept, int num_finds )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->load_and_find(
			file_name, image, width, height,
			x, y, correlation, ambig, ngc,
			search_center_x, search_center_y, search_width, search_height,
			time_out, y_origin_ll, min_accept, max_accept, num_finds );
	return retVal;
}

int vswrapper::load_finder_template( const char *file_name )
{ 
	int retVal=0;
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->load_finder_template( file_name );
	return retVal;
}

// performs vsfind given template object.
const char * vswrapper::find_from_template(
	unsigned char *image, int width, int height,
	double &x, double &y, double &correlation, double &ambig, double *ngc,
	double search_center_x, double search_center_y, double search_width, double search_height,
	double time_out, int y_origin_ll, double min_accept, double max_accept, int num_finds )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->find_from_template(
			image, width, height,
			x, y, correlation, ambig, ngc,
			search_center_x, search_center_y, search_width, search_height,
			time_out, y_origin_ll, min_accept, max_accept, num_finds );
	return retVal;
}
*/
VsEnviron& vswrapper::getStaticEnv()
{
	static VsEnviron retVal = 0;
	return retVal;
}

DWORD& vswrapper::getEnvThread()
{
	static DWORD retVal = 0;
	return retVal;
}

int vswrapper::getAddEnvUseCount( int add/*=0*/ )
{
	::CRITICAL_SECTION cs;
	::InitializeCriticalSection(&cs);
	::EnterCriticalSection(&cs);

	// int with a default initialization to 0
	struct UseNum { UseNum() : num(0) {} int num; }; 
	static std::map<DWORD,UseNum> useCount;

	useCount[::GetCurrentThreadId()].num += add;
	
	int retVal = useCount[::GetCurrentThreadId()].num;
	
	::LeaveCriticalSection(&cs);
	::DeleteCriticalSection(&cs);
	return retVal;
}


VsEnviron vswrapper::getEnv()
{
	VsEnviron& retVal = getStaticEnv();

	if (!retVal)
	{
		retVal = ::vsCreateSharedVisionEnviron(0, 0);
		getEnvThread() = ::GetCurrentThreadId();
	}
	
	DWORD envThread = getEnvThread();

	if (envThread != ::GetCurrentThreadId() )
	{
		int useCount = getAddEnvUseCount(+1);
		if (useCount == 1)
		{
			if(vsAttachThread() == -1)
				throw std::runtime_error("vsAttachThread() failed");
		}
	}
	return retVal;
}


void vswrapper::releaseEnv()
{
	DWORD envThread = getEnvThread();

	if (envThread != ::GetCurrentThreadId() )
	{
		int useCount = getAddEnvUseCount(-1);
		if( useCount == 0 )
		{
			if (vsDetachThread(NULL) == -1)
				throw std::runtime_error("vsDetachThread() failed");
		}
		else if( useCount < 0 ) 
		{
			throw std::runtime_error("vsDetachThread() called when already detached");
		}
	}
}

void vswrapper::disposeEnv()
{
	VsEnviron& env = getStaticEnv();
	DWORD envThread = getEnvThread();

	if (env && envThread == ::GetCurrentThreadId() )
	{
		::vsDispose(env);
		env = 0;
	}
}

///////////////////////////////////////////////////////////////////////////////
// Modified version for for 2D-SPI application
//////////////////////////////////////////////////////////////////////////////

const char * vswrapper::create_disc_template(
	int* piNodeID,			// Output: nodeID of map	
	templatetype tpl, double r, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_disc_template(
			piNodeID,
			tpl, r, theta, 
			dark_to_light, min_scale, max_scale, 
			low_accept, high_accept, mask_region, depth );

	return retVal;
}

// creates a template for locating a cross
const char * vswrapper::create_cross_template(
	int* piNodeID,			// Output: nodeID of map	
	templatetype tpl, double base, double height, 
	double base_leg, double height_leg, double theta, int dark_to_light,
	double *min_scale, double *max_scale, int rounded_edges, 
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_cross_template(
			piNodeID, 
			tpl, base, height,
			base_leg, height_leg, rounded_edges,
			theta, dark_to_light,
			min_scale, max_scale, 
			low_accept, high_accept, mask_region, depth );

	return retVal;
}


// creates a template for locating a cross
const char * vswrapper::create_diamond_template(
	int* piNodeID,			// Output: nodeID of map	
	templatetype tpl, double base, double height, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_diamond_template(
			piNodeID,
			tpl, base, height, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );

	return retVal;
}

// creates a template for locating a cross
const char * vswrapper::create_donut_template(
	int* piNodeID,			// Output: nodeID of map	
	templatetype tpl, double inner_radius, double outer_radius, double theta, 
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_donut_template(
			piNodeID,
			tpl, inner_radius, outer_radius, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );

	return retVal;
}


const char * vswrapper::create_triangle_template(
	int* piNodeID,			// Output: nodeID of map	
	templatetype tpl, double base, double height, double offset, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_triangle_template(
			piNodeID,
			tpl, base, height, offset, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );

	return retVal;
}


const char * vswrapper::create_rectangle_template(
	int* piNodeID,			// Output: nodeID of map	
	templatetype tpl, double base, double height, double theta,
	int dark_to_light, double *min_scale, double *max_scale,
	double low_accept, double high_accept, double mask_region, int depth )
{ 
	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->create_rectangle_template(
			piNodeID, 
			tpl, base, height, theta,
			dark_to_light, min_scale, max_scale,
			low_accept, high_accept, mask_region, depth );

	return retVal;
}

// load the template by a given name and find the match in the given image.
const char * vswrapper::Find(
	int iNodeID,				
	unsigned char *image, int width, int height,
	double &x, double &y, double &correlation, double &ambig, double *ngc,
	double search_center_x, double search_center_y, double search_width, double search_height,
	double time_out, int y_origin_ll, double min_accept, double max_accept, int num_finds )
{ 

	const char* retVal="";
	if( concreteWrapper_ ) 
		retVal = concreteWrapper_->Find(
			iNodeID,	
			image, width, height,
			x, y, correlation, ambig, ngc,
			search_center_x, search_center_y, search_width, search_height,
			time_out, y_origin_ll, min_accept, max_accept, num_finds );

	return retVal;
}


