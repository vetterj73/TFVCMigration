#pragma once

#include "Panel.h"
#include "Image.h"
#include "UIRect.h"

class NgcTemplateSt;
class CyberNgcFiducialCorrelation
{
public:
	CyberNgcFiducialCorrelation(void);
	~CyberNgcFiducialCorrelation(void);

	int CreateNgcTemplate(Feature* pFid, const Image* pTemplateImg, UIRect tempRoi);

	// load the template by a given name and find the match in the given image.
	bool Find(
		int iNodeID,			// map ID of template  and finder		
		Image* pSearchImage,	// buffer containing the image
		UIRect searchRoi,		// width of the image in pixels
		double &x,              // returned x location of the center of the template from the origin
		double &y,              // returned x location of the center of the template from the origin
		double &correlation,    // match score 0-1
		double &ambig);			// ratio of (second best/best match) score 0-1

protected:
	bool CreateNgcTemplate(Feature* pFid, const Image* pTemplateImg, UIRect tempRoi, int* pTemplateID);
	int GetNgcTemplateID(Feature* pFeature);
	unsigned int CalculateRingHalfWidth(Feature* pFid, double dImageResolution);

private:
	list<NgcTemplateSt> _ngcTemplateStList;
	unsigned int _iCurrentIndex;

	unsigned int _iDepth;	// Depth for template creation and searching
};

