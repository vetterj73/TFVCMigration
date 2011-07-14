#include "CyberNgcFiducialCorrelation.h"
#include "Ngc.h"

// Class for internal use only
class NgcTemplateSt
{
public:
	NgcTemplateSt()
	{
		_iTemplateID = -1;
		_ptTemplate = NULL;
		_bValid = false;
	};

	~NgcTemplateSt()
	{
	};

	Feature* _pFeature;
	VsStCTemplate* _ptTemplate;
	unsigned int _iTemplateID;
	bool _bValid;
};


CyberNgcFiducialCorrelation::CyberNgcFiducialCorrelation(void)
{
	_iCurrentIndex = 0;
}


CyberNgcFiducialCorrelation::~CyberNgcFiducialCorrelation(void)
{
	list<NgcTemplateSt>::iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(i->_bValid)
		{
			vsDispose2DTemplate(i->_ptTemplate);
			if(i->_ptTemplate!=NULL) delete i->_ptTemplate;
		}
	}

	_ngcTemplateStList.clear();
}

// Create Ngc template for a fiducial if it doesn't exist
// pFid: input, fiducial feature
// pTemplateImg and tempRect: input, tempate image and rectangle
// Return template ID for existing or new create ngc template
// Return -1 if failed
int CyberNgcFiducialCorrelation::CreateNgcTemplate(Feature* pFid, const Image* pTemplateImg, UIRect tempRoi)
{
	// If the template exists
	int iTemplateID = GetNgcTemplateID(pFid);
	if(iTemplateID >= 0) return(iTemplateID);
	
	// Create a new template
	bool bFlag = CreateNgcTemplate(pFid,  pTemplateImg, tempRoi, &iTemplateID);
	if(!bFlag) return(-1);

	return(iTemplateID);
}

// Find a match in the search image and report results
bool CyberNgcFiducialCorrelation::Find(
		int iNodeID,			// map ID of template  and finder		
		Image* pSearchImage,   // buffer containing the image
		UIRect searchRoi,      // width of the image in pixels
		double &x,              // returned x location of the center of the template from the origin
		double &y,              // returned x location of the center of the template from the origin
		double &correlation,    // match score [-1,1]
		double &ambig)          // ratio of (second best/best match) score 0-1
{
	// Get the template from the list
	VsStCTemplate* ptTemplate = NULL;
	list<NgcTemplateSt>::iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(i->_iTemplateID == iNodeID)
		{
			ptTemplate = i->_ptTemplate;
			break;
		}
	}

	if(ptTemplate == NULL)
		return(false);

	// Search
	SvImage oSearchImage;
	oSearchImage.pdData = pSearchImage->GetBuffer();
	oSearchImage.iWidth = pSearchImage->Columns();
	oSearchImage.iHeight = pSearchImage->Rows();
	oSearchImage.iSpan = pSearchImage->PixelRowStride();
    
	VsStRect searchRect;
	searchRect.lXMin = searchRoi.FirstColumn;
	searchRect.lXMax = searchRoi.LastColumn;
	searchRect.lYMin = searchRoi.FirstRow;
	searchRect.lYMax = searchRoi.LastRow;
	
	// Set correlation paramter	
	VsStCorrelate tCorrelate;
	//tCorrelate.dGainTolerance			= 0.99;	// This is ridiciously high, but seems working in this way
	tCorrelate.dLoResMinScore			= 0.25;	// Intentionally low these two value for ambiguous check
    tCorrelate.dHiResMinScore			= 0.25;
    tCorrelate.iMaxResultPoints			= 2;
	// Flat peak check
	//tCorrelate.dFlatPeakThreshPercent	= 4.0 /* CORR_AREA_FLAT_PEAK_THRESH_PERCENT */;
    //tCorrelate.iFlatPeakRadiusThresh	= 5;

	unsigned int iDepth = 3;

	int iFlag =vs2DCorrelate(
		ptTemplate, &oSearchImage, 
		searchRect, iDepth, &tCorrelate);
	if(iFlag < 0) // Error or no match
	{	
		vsDispose2DCorrelate(&tCorrelate);	
		return(false);
	}
	if(tCorrelate.iNumResultPoints == 0) // No Match 
	{	
		vsDispose2DCorrelate(&tCorrelate);
		return(false);
	}
	if(fabs(tCorrelate.ptCPoint[0].dScore) < 0.4) // Match is too low
	{	
		vsDispose2DCorrelate(&tCorrelate);
		return(false);
	}

	// Get results
	x = tCorrelate.ptCPoint[0].dLoc[0]; 
	y = tCorrelate.ptCPoint[0].dLoc[1];
	correlation = tCorrelate.ptCPoint[0].dScore;
	if(tCorrelate.iNumResultPoints >=2)
		ambig= fabs(tCorrelate.ptCPoint[1].dScore/tCorrelate.ptCPoint[0].dScore);
	else
		ambig = 0;

	// Clean up
	vsDispose2DCorrelate(&tCorrelate);

	return(true);
}


// Create ngc template for a fiducial
// pFid: input, fiducial feature
// pTemplateImg and tempRect: input, tempate image and rectangle
// pTemplateID: output, ID of vsfinder template
bool CyberNgcFiducialCorrelation::CreateNgcTemplate(Feature* pFid, const Image* pTemplateImg, UIRect tempRoi, int* pTemplateID)
{
	// Create template
	SvImage oTempImage;
	oTempImage.pdData = pTemplateImg->GetBuffer();
	oTempImage.iWidth = pTemplateImg->Columns();
	oTempImage.iHeight = pTemplateImg->Rows();
	oTempImage.iSpan = pTemplateImg->PixelRowStride();
	
	VsStRect templateRect;
	templateRect.lXMin = tempRoi.FirstColumn;
	templateRect.lXMax = tempRoi.LastColumn;
	templateRect.lYMin = tempRoi.FirstRow;
	templateRect.lYMax = tempRoi.LastRow;
	
	int iDepth = 3;
	
	NgcTemplateSt templateSt;
	templateSt._ptTemplate = new VsStCTemplate();
	int iFlag = vsCreate2DTemplate(&oTempImage, templateRect, iDepth, templateSt._ptTemplate);
	if(iFlag<0)
	{
		delete templateSt._ptTemplate;
		*pTemplateID = -1; 
		return(false);
	}

	// Fill the struct
	templateSt._bValid = true;
	templateSt._iTemplateID = _iCurrentIndex;
	_iCurrentIndex++;
	templateSt._pFeature = pFid;

	// Add to the list
	_ngcTemplateStList.push_back(templateSt);
	*pTemplateID = templateSt._iTemplateID;

	return(true);
}

// Return the template ID for a feature if a template for it already exists
// otherwise, return -1
int CyberNgcFiducialCorrelation::GetNgcTemplateID(Feature* pFeature)
{
	list<NgcTemplateSt>::const_iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(pFeature->GetShape() == i->_pFeature->GetShape()) // Type check
		{
			switch(pFeature->GetShape())// Fearure content check
			{
			case Feature::SHAPE_CROSS:
				{
					CrossFeature* pFeature1 = (CrossFeature*)pFeature;
					CrossFeature* pFeature2 = (CrossFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY() &&
						pFeature1->GetLegSizeX() == pFeature2->GetLegSizeX() &&
						pFeature1->GetLegSizeY() == pFeature2->GetLegSizeY())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_DIAMOND:
				{					
					DiamondFeature* pFeature1 = (DiamondFeature*)pFeature;
					DiamondFeature* pFeature2 = (DiamondFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_DISC:
				{
					DiscFeature* pFeature1 = (DiscFeature*)pFeature;
					DiscFeature* pFeature2 = (DiscFeature*)i->_pFeature;

					if(pFeature1->GetDiameter() == pFeature2->GetDiameter())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_DONUT:
				{
					DonutFeature* pFeature1 = (DonutFeature*)pFeature;
					DonutFeature* pFeature2 = (DonutFeature*)i->_pFeature;

					if(pFeature1->GetDiameterOutside() == pFeature2->GetDiameterOutside() &&
						pFeature1->GetDiameterInside() == pFeature2->GetDiameterInside())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_RECTANGLE:
				{
					RectangularFeature* pFeature1 = (RectangularFeature*)pFeature;
					RectangularFeature* pFeature2 = (RectangularFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY())
						return(i->_iTemplateID);
				}
				break;

			case Feature::SHAPE_TRIANGLE:
				{
					TriangleFeature* pFeature1 = (TriangleFeature*)pFeature;
					TriangleFeature* pFeature2 = (TriangleFeature*)i->_pFeature;

					if(pFeature1->GetSizeX() == pFeature2->GetSizeX() &&
						pFeature1->GetSizeY() == pFeature2->GetSizeY() &&
						pFeature1->GetOffset() == pFeature2->GetOffset())
						return(i->_iTemplateID);
				}
				break;

			default:
				break;
			}
		}
	}

	return(-1);
}
