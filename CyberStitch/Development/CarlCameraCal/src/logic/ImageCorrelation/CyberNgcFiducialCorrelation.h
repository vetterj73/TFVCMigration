#pragma once

#include "Panel.h"
#include "Image.h"
#include "UIRect.h"
#include "windows.h"

class NgcTemplateSt;
class CyberNgcFiducialCorrelation
{
protected:
	CyberNgcFiducialCorrelation(void);
	~CyberNgcFiducialCorrelation(void);

	static CyberNgcFiducialCorrelation* pInstance;

public:
	static CyberNgcFiducialCorrelation& Instance();

	int CreateNgcTemplate(
		Feature* pFid,
		bool bFidBrighterThanBackground,
		bool bFiducialAllowNegativeMatch,
		const Image* pTemplateImg, // Always Fiducial is brighter than background in image
		UIRect tempRoi);

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
	bool CreateNgcTemplate(
		Feature* pFid, 
		bool bCreatePositiveTemplate,
		bool bCreateNegativeTemplate,
		const Image* pTemplateImg, 
		UIRect tempRoi, 
		int* pTemplateID);
	int GetNgcTemplateID(
		Feature* pFeature,
		bool bHasPositiveTemplate,
		bool bHasNegativeTemplate);
	unsigned int CalculateRingHalfWidth(Feature* pFid, double dImageResolution);

private:
	list<NgcTemplateSt> _ngcTemplateStList;
	unsigned int _iCurrentIndex;

	unsigned int _iDepth;	// Depth for template creation and searching

	// Protect template creation stage
	HANDLE _entryMutex; 
};

