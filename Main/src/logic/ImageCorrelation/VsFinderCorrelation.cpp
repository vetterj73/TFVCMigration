#include "VsFinderCorrelation.h"
#include "vswrapper.h"
#include "Logger.h"

// For singleton pattern
VsFinderCorrelation* VsFinderCorrelation::pInstance = 0;
VsFinderCorrelation& VsFinderCorrelation::Instance()
{
	if( pInstance == NULL )
		pInstance = new VsFinderCorrelation();

	return *pInstance;
}

VsFinderCorrelation::VsFinderCorrelation()
{
	_pVsw = new vswrapper();

	_entryMutex = CreateMutex(0, FALSE, "Entry Mutex"); // No initial owner
}

void VsFinderCorrelation::Config(double dResolution, unsigned int iColumn, unsigned int iRows)
{
	_pVsw->set_pixel_size(dResolution);
	_pVsw->set_fov_cols(iColumn);
	_pVsw->set_fov_rows(iRows);
}

VsFinderCorrelation::~VsFinderCorrelation()
{
	CloseHandle(_entryMutex);
	delete _pVsw;
}

// Create vsfinder template for a fiducial if it doesn't exist
// pFid: input, fiducial feature
// Return template ID for existing or new create vsfinder template
// Return -1 if failed
int VsFinderCorrelation::CreateVsTemplate(
	Feature* pFid, 
	bool bFiducialBrighterThanBackground, 
	bool bFiducialAllowNegativeMatch)
{
	// Mutex protection
	WaitForSingleObject(_entryMutex, INFINITE);

	// If the template exists
	int iTemplateID = GetVsTemplateID(pFid, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch);
	if(iTemplateID >= 0) 
	{
		ReleaseMutex(_entryMutex);
		return(iTemplateID);
	}
	
	// Create a new template
	bool bFlag = CreateVsTemplate(pFid, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch, &iTemplateID);
	if(!bFlag) 
	{
		ReleaseMutex(_entryMutex);
		return(-1);
	}

	// Add new template into list
	FeatureTemplateID templateID(pFid, bFiducialBrighterThanBackground, bFiducialAllowNegativeMatch, iTemplateID);
	_vsTemplateIDList.push_back(templateID);
		
	// Mutex protection
	ReleaseMutex(_entryMutex);
	
	return(iTemplateID);
}

// load the template by a given name and find the match in the given image.
void VsFinderCorrelation::Find(
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
		int y_origin_ll,      // If !0 origin in start of the image is 0,0 increasing in Y and X as you move forward
		double min_accept,   // If >0 minimum score to persue at min pyramid level to look for peak override
		double max_accept,   // If >0 minumum score to accept at max pyramid level to look for peak override
		int num_finds        // If > 0 number of versions of the template to find
	)
{
	// Mutex protection
	WaitForSingleObject(_entryMutex, INFINITE);

	_pVsw->Find(
		iNodeID,			// map ID of template  and finder		
		image,				// buffer containing the image
		width,              // width of the image in pixels
		height,             // height of the image in pixels
		x,					// returned x location of the center of the template from the origin
		y,					// returned x location of the center of the template from the origin
		correlation,		// match score 0-1
		ambig,				// ratio of (second best/best match) score 0-1
		ngc,				// Normalized Grayscale Correlation Score 0-1
		search_center_x,	// x center of the search region in pixels
		search_center_y,	// y center of the search region in pixels
		search_width,		// width of the search region in pixels
		search_height,		// height of the search region in pixels
		time_out,			// number of seconds to search maximum. If limit is reached before any results found, an error will be generated
		y_origin_ll,		// If !0 origin in start of the image is 0,0 increasing in Y and X as you move forward
		min_accept,			// If >0 minimum score to persue at min pyramid level to look for peak override
		max_accept,			// If >0 minumum score to accept at max pyramid level to look for peak override
		num_finds);			// If > 0 number of versions of the template to find

	// Mutex protection
	ReleaseMutex(_entryMutex);
}

// Create vsfinder template for a fiducial
// pFid: input, fiducial feature
// pTemplateID: output, ID of vsfinder template
bool VsFinderCorrelation::CreateVsTemplate(
	Feature* pFid,
	bool bFiducialBrighterThanBackground, 
	bool bFiducialAllowNegativeMatch,
	int* pTemplateID)
{
	double dMinScale[2]={0.95, 0.95};
	double dMaxScale[2]={1.05, 1.05};

	int iFidBrighterThanBackground = bFiducialBrighterThanBackground? 1 : 0;

	// Convert roation angle from count-clockwise degreee to clockwise and unit 1 for 360 degree
	double dTheta = -(pFid->GetRotation()-90.)/360.; 
	switch(pFid->GetShape())
	{
	case Feature::SHAPE_CROSS:
		_pVsw->create_cross_template(pTemplateID, vswrapper::FIDUCIAL,
			((CrossFeature*)pFid)->GetSizeX(), ((CrossFeature*)pFid)->GetSizeY(),
			((CrossFeature*)pFid)->GetLegSizeX(), ((CrossFeature*)pFid)->GetLegSizeY(),
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_DIAMOND:
		_pVsw->create_diamond_template(pTemplateID, vswrapper::FIDUCIAL,
			((DiamondFeature*)pFid)->GetSizeX(), ((DiamondFeature*)pFid)->GetSizeY(),
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_DIAMONDFRAME:
		_pVsw->create_diamondframe_template(pTemplateID, vswrapper::FIDUCIAL,
			((DiamondFeature*)pFid)->GetSizeX(), ((DiamondFeature*)pFid)->GetSizeY(), ((DiamondFrameFeature*)pFid)->GetThickness(),
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_DISC:
		_pVsw->create_disc_template(pTemplateID, vswrapper::FIDUCIAL,
			((DiscFeature*)pFid)->GetDiameter()/2,
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_DONUT:
		_pVsw->create_donut_template(pTemplateID, vswrapper::FIDUCIAL,
			((DonutFeature*)pFid)->GetDiameterInside()/2,((DonutFeature*)pFid)->GetDiameterOutside()/2,
			dTheta, 1, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_RECTANGLE:
		_pVsw->create_rectangle_template(pTemplateID, vswrapper::FIDUCIAL,
			((RectangularFeature*)pFid)->GetSizeX(), ((RectangularFeature*)pFid)->GetSizeY(),
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_RECTANGLEFRAME:
		_pVsw->create_rectangleframe_template(pTemplateID, vswrapper::FIDUCIAL,
			((RectangularFeature*)pFid)->GetSizeX(), ((RectangularFeature*)pFid)->GetSizeY(), ((RectangularFrameFeature*)pFid)->GetThickness(),
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_TRIANGLE:
		_pVsw->create_triangle_template(pTemplateID, vswrapper::FIDUCIAL,
			((TriangleFeature*)pFid)->GetSizeX(), ((TriangleFeature*)pFid)->GetSizeY(),
			((TriangleFeature*)pFid)->GetOffset(),
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_EQUILATERALTRIANGLEFRAME:
		_pVsw->create_triangleframe_template(pTemplateID, vswrapper::FIDUCIAL,
			((TriangleFeature*)pFid)->GetSizeX(), ((TriangleFeature*)pFid)->GetSizeY(), 
			((TriangleFeature*)pFid)->GetOffset(), ((EquilateralTriangleFrameFeature*)pFid)->GetThickness(),
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_CHECKERPATTERN:
		_pVsw->create_checkerpattern_template(pTemplateID, vswrapper::FIDUCIAL,
			((CheckerPatternFeature*)pFid)->GetSizeX(), ((CheckerPatternFeature*)pFid)->GetSizeY(), 
			dTheta, iFidBrighterThanBackground, bFiducialAllowNegativeMatch, dMinScale, dMaxScale);
		break;

	case Feature::SHAPE_CYBER:
		LOG.FireLogEntry(LogTypeError, "CreateVsTemplate():unsupported fiducial type");
		break;

	default:
		LOG.FireLogEntry(LogTypeError, "CreateVsTemplate():unsupported fiducial type");	
	}

	return(true);
}

// Return the template ID for a feature if a template for it already exists
// otherwise, return -1
int VsFinderCorrelation::GetVsTemplateID(
	Feature* pFeature,
	bool bFidBrighterThanBackground,
	bool bFiducialAllowNegativeMatch)
{
	list<FeatureTemplateID>::const_iterator i;
	for(i=_vsTemplateIDList.begin(); i!=_vsTemplateIDList.end(); i++)
	{
		if(i->_bFidBrighterThanBackground == bFidBrighterThanBackground &&
			i->_bFiducialAllowNegativeMatch == bFiducialAllowNegativeMatch &&
			IsSameTypeSize(pFeature, i->_pFeature))
				return(i->_iTemplateID);
	}

	return(-1);
}