#include "FeatureLocationCheck.h"
#include "Feature.h"
#include "Image.h"
#include "RenderShape.h"
#include "VsFinderCorrelation.h"
#include <map>
using std::map;
typedef map<int, Feature*> FeatureList;
typedef FeatureList::iterator FeatureListIterator;


// fiducial is placed in the exact center of the resulting image
// this is important as SqRtCorrelation (used to build least squares table)
// measures the positional difference between the center of the fiducial 
// image and the center of the FOV image segement.  Any information
// about offset of the fiducial from the center of the FOV is lost
// img: output, fiducial image,
// fid: input, fiducial feature 
// resolution: the pixel size, in meters
// dScale: the fiducial area expansion scale
void RenderFiducial(
	Image* pImg, 
	Feature* pFid, 
	double resolution, 
	double dScale)
{
	// Area in world space
	double dExpandX = 0.005;
	double dExpandY = 0.005;
	double dHalfWidth = pFid->GetBoundingBox().Width()/2;
	double dHalfHeight= pFid->GetBoundingBox().Height()/2;
	double xMinImg = pFid->GetBoundingBox().Min().x - dHalfWidth*(dScale-1)  - dExpandX; 
	double yMinImg = pFid->GetBoundingBox().Min().y - dHalfHeight*(dScale-1) - dExpandY; 
	double xMaxImg = pFid->GetBoundingBox().Max().x + dHalfWidth*(dScale-1)  + dExpandX; 
	double yMaxImg = pFid->GetBoundingBox().Max().y + dHalfHeight*(dScale-1) + dExpandY;
	
	// The size of image in pixels
	unsigned int nRowsImg
			(unsigned int((1.0/resolution)*(xMaxImg-xMinImg)));	
	unsigned int nColsImg
			(unsigned int((1.0/resolution)*(yMaxImg-yMinImg)));	
	
	// Define transform for new fiducial image
	double t[3][3];
	t[0][0] = resolution;
	t[0][1] = 0;
	t[0][2] = pFid->GetCadX() - resolution * (nRowsImg-1) / 2.;
	
	t[1][0] = 0;
	t[1][1] = resolution;
	t[1][2] = pFid->GetCadY() - resolution * (nColsImg-1) / 2.;
	
	t[2][0] = 0;
	t[2][1] = 0;
	t[2][2] = 1;	
	
	ImgTransform trans(t);

	// Create and clean buffer
	pImg->Configure(nColsImg, nRowsImg, nColsImg, trans, trans, true);
	pImg->ZeroBuffer();

	// Fiducial drawing/render
	unsigned int grayValue = 255;
	int antiAlias = 1;
	RenderFeature(pImg, pFid, grayValue, antiAlias);
}


// Create Fiducial images
void CreateFiducialImage(Panel* pPanel, Image* pImage, Feature* pFeature)
{
	double dScale = 1.4;
	
	RenderFiducial(
		pImage, 
		pFeature, 
		pPanel->GetPixelSizeX(), 
		dScale);	
}

bool VsfinderAlign(
	Image* pImg,
	int iTemplateID,
	double dSearchColCen,
	double dSearchRowCen,
	double dSearchWidth,
	double dSearchHeight,
	double* pdScore,
	double* pdAmbig,
	double* pdX,
	double* pdY
	)
{
	double x, y, corscore, ambig, ngc;
	double time_out = 1e5;			// MicroSeconds
	double dMinScore = 0.5;
	
	VsFinderCorrelation::Instance().Find(
		iTemplateID,		// map ID of template  and finder
		pImg->GetBuffer(),	// buffer containing the image
		pImg->Columns(),	// width of the image in pixels
		pImg->Rows(),		// height of the image in pixels
		x,					// returned x location of the center of the template from the origin
		y,					// returned x location of the center of the template from the origin
		corscore,			// match score 0-1
		ambig,				// ratio of (second best/best match) score 0-1
		&ngc,				// Normalized Grayscale Correlation Score 0-1
		dSearchColCen,		// Col center of the search region in pixels
		dSearchRowCen,		// Row center of the search region in pixels
		dSearchWidth,		// width of the search region in pixels
		dSearchHeight,		// height of the search region in pixels
		time_out,			// number of seconds to search maximum. If limit is reached before any results found, an error will be generated	
		0,					// image origin is top-left				
		dMinScore/3,		// If >0 minimum score to persue at min pyramid level to look for peak override
		dMinScore/3);		// If >0 minumum score to accept at max pyramid level to look for peak override
							// Use a lower minimum score for vsfinder so that we can get a reliable ambig score

	if(corscore > dMinScore)	// Valid results
	{
		*pdScore = corscore;
		*pdAmbig = ambig;
		*pdX = x;
		*pdY = y;
	}
	else	// Invalid results
	{
		*pdScore = 0;
		*pdAmbig = 1;
		*pdX = -1;
		*pdY = -1;
	}

	return(true);
}

// pPanel: the point for fidcial descrptions
FeatureLocationCheck::FeatureLocationCheck(Panel* pPanel)
{
	_pPanel = pPanel;
	int iNum = _pPanel->NumberOfFiducials();
	_piTemplateIds = new int[iNum];

	// Vsfinder initialize
	VsFinderCorrelation::Instance().Config(
		_pPanel->GetPixelSizeX(), _pPanel->GetNumPixelsInY(), _pPanel->GetNumPixelsInX());

	int iCount = 0;
	for(FeatureListIterator iFid = _pPanel->beginFiducials(); iFid != _pPanel->endFiducials(); iFid++)
	{
		// Create a fiducial image
		Image fiducialImg;
		CreateFiducialImage(_pPanel, &fiducialImg, iFid->second);
		bool bFidBrighter = true;
		bool bAllowNegMatch = false;

		// Create Fiducial template 
		int iVsFinderTemplateId = VsFinderCorrelation::Instance().CreateVsTemplate(iFid->second, bFidBrighter, bAllowNegMatch);
		if(iVsFinderTemplateId < 0) 
			return;

		_piTemplateIds[iCount] = iVsFinderTemplateId;
		iCount++;
	}
}


FeatureLocationCheck::~FeatureLocationCheck(void)
{
	delete [] _piTemplateIds;
}

// Find locations of fiducials in a panel image
// pImage: input, panel image
// dResults: output, the fidcuial results,
// for each fiducial return(cad_x, cad_y, Loc_x, loc_y, corrScore, ambig)
bool FeatureLocationCheck::CheckFeatureLocation(Image* pImage, double dResults[])
{	
	// Setttings
	double dSearchExpansion = 5e-3; // 5 mm
	int iItems = 6;
	
	if(pImage == NULL) return(false);
	double dPixelSize = pImage->PixelSizeX();

	// For each fiducial
	int iCount = 0;
	for(FeatureListIterator iFid = _pPanel->beginFiducials(); iFid != _pPanel->endFiducials(); iFid++)
	{
		// Prepare search area
		Box box = iFid->second->GetBoundingBox();
		double dSearchRowCen, dSearchColCen;
		pImage->WorldToImage(box.Center().x, box.Center().y, &dSearchRowCen, &dSearchColCen);
		double dSearchHeight = (box.Width() + dSearchExpansion*2)/dPixelSize; // The CAD and image has 90 degree rotation
		double dSearchWidth = (box.Height() + dSearchExpansion*2)/dPixelSize;

		// Find fiducial
		double dCol, dRow, dScore, dAmbig; 
		bool bFlag= VsfinderAlign(
			pImage,
			_piTemplateIds[iCount],
			dSearchColCen,
			dSearchRowCen,
			dSearchWidth,
			dSearchHeight,
			&dScore,
			&dAmbig,
			&dCol,
			&dRow);

		double dX, dY;
		if(dScore==0 || dAmbig==1) // If feature/fiducial is not found
		{
			dX = -1;
			dY = -1;
		}
		else
			pImage->ImageToWorld(dRow, dCol, &dX, &dY);

		// Record results
		dResults[iCount*iItems] = box.Center().x;	// CAD x  
		dResults[iCount*iItems+1] = box.Center().y; // CAD y
		dResults[iCount*iItems+2] = dX;				// Loc x  
		dResults[iCount*iItems+3] = dY;				// Loc y
		dResults[iCount*iItems+4] = dScore;			// Loc x  
		dResults[iCount*iItems+5] = dAmbig;			// Loc y

		iCount++;
	}

	return(true);
}
