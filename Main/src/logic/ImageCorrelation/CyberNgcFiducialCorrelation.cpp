#include "CyberNgcFiducialCorrelation.h"
#include "Ngc.h"
#include "Utilities.h"
#include "morpho.h"
#include "CorrelationParameters.h"


// Class for internal use only
class NgcTemplateSt
{
public:
	NgcTemplateSt()
	{
		_iTemplateID = -1;
		_ptTemplate = NULL;
		_ptNegTemplate = NULL;
		_bValid = false;
	};

	~NgcTemplateSt()
	{
	};

	Feature* _pFeature;
	VsStCTemplate* _ptTemplate;
	VsStCTemplate* _ptNegTemplate;
	unsigned int _iTemplateID;
	bool _bValid;
};


CyberNgcFiducialCorrelation::CyberNgcFiducialCorrelation(void)
{
	_iCurrentIndex = 0;

	_iDepth = 3; 
}


CyberNgcFiducialCorrelation::~CyberNgcFiducialCorrelation(void)
{
	list<NgcTemplateSt>::iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(i->_bValid)
		{
			if(i->_ptTemplate!=NULL) 
			{
				vsDispose2DTemplate(i->_ptTemplate);
				delete i->_ptTemplate;
			}
			if(i->_ptNegTemplate!=NULL)
			{
				vsDispose2DTemplate(i->_ptNegTemplate);
				delete i->_ptNegTemplate;
			}
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
	double dMinScore = CorrelationParametersInst.dCyberNgcMinCorrScore;

	// Get the template from the list
	VsStCTemplate* ptTemplate = NULL;
	VsStCTemplate* ptNegTemplate = NULL;
	list<NgcTemplateSt>::iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(i->_iTemplateID == iNodeID)
		{
			ptTemplate = i->_ptTemplate;
			ptNegTemplate = i->_ptNegTemplate;
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
	tCorrelate.dGainTolerance			= 5;	
	tCorrelate.dLoResMinScore			= dMinScore/2;	// Intentionally low these two value for ambiguous check
    tCorrelate.dHiResMinScore			= dMinScore/2;
    tCorrelate.iMaxResultPoints			= 2;
	// Flat peak check
	//tCorrelate.dFlatPeakThreshPercent	= 4.0 /* CORR_AREA_FLAT_PEAK_THRESH_PERCENT */;
    //tCorrelate.iFlatPeakRadiusThresh	= 5;

	double x1, x2, y1, y2, corr1, corr2, ambig1, ambig2;
	bool bSuccess1=true, bSuccess2=true;

	// Try postive template
	int iFlag =vs2DCorrelate(
		ptTemplate, &oSearchImage, 
		searchRect, _iDepth, &tCorrelate);
	if(iFlag < 0 || tCorrelate.iNumResultPoints == 0 || fabs(tCorrelate.ptCPoint[0].dScore) < dMinScore) // Error or no match
	{	
		bSuccess1= false;
		vsDispose2DCorrelate(&tCorrelate);
	}
	else
	{
		// Get results
		x1 = tCorrelate.ptCPoint[0].dLoc[0]; 
		y1 = tCorrelate.ptCPoint[0].dLoc[1];
		corr1 = tCorrelate.ptCPoint[0].dScore;
		if(tCorrelate.iNumResultPoints >=2)
			ambig1= fabs(tCorrelate.ptCPoint[1].dScore/tCorrelate.ptCPoint[0].dScore);
		else
			ambig1 = 0;

		vsDispose2DCorrelate(&tCorrelate);
	
		if(corr1 > 0.7  &&ambig1 < 0.5 )	// if it is good enough
		{
			x = x1;
			y = y1;
			correlation = corr1;
			ambig = ambig1;
			return(true);
		}	
	}

	// Try negative template
	iFlag =vs2DCorrelate(
		ptNegTemplate, &oSearchImage, 
		searchRect, _iDepth, &tCorrelate);
	if(iFlag < 0 || tCorrelate.iNumResultPoints == 0 || fabs(tCorrelate.ptCPoint[0].dScore) < dMinScore) // Error or no match
	{	
		bSuccess2= false;
		vsDispose2DCorrelate(&tCorrelate);
	}
	else
	{
		// Get results
		x2 = tCorrelate.ptCPoint[0].dLoc[0]; 
		y2 = tCorrelate.ptCPoint[0].dLoc[1];
		corr2 = tCorrelate.ptCPoint[0].dScore;
		if(tCorrelate.iNumResultPoints >=2)
			ambig2= fabs(tCorrelate.ptCPoint[1].dScore/tCorrelate.ptCPoint[0].dScore);
		else
			ambig2 = 0;

		vsDispose2DCorrelate(&tCorrelate);
	}

	// Decision logic
	if(!bSuccess1 && !bSuccess2)		// None of them
		return(false);
	else if( bSuccess1 && !bSuccess2)	// The first one
	{
		x = x1;
		y = y1;
		correlation = corr1;
		ambig = ambig1;
	}
	else if( !bSuccess1 && bSuccess2)	// The second one
	{
		x = x2;
		y = y2;
		correlation = corr2;
		ambig = ambig2;
	}
	else if( bSuccess1 && bSuccess2)	// Pick the better one
	{
		if(corr1*(1-ambig1) > corr2*(1-ambig2))
		{
			x = x1;
			y = y1;
			correlation = corr1;
			ambig = ambig1;
		}
		else
		{
			x = x2;
			y = y2;
			correlation = corr2;
			ambig = ambig2;
		}
	}

	return(true);
}


// Create ngc template for a fiducial
// pFid: input, fiducial feature
// pTemplateImg and tempRect: input, tempate image and rectangle
// pTemplateID: output, ID of vsfinder template
bool CyberNgcFiducialCorrelation::CreateNgcTemplate(Feature* pFid, const Image* pTemplateImg, UIRect tempRoi, int* pTemplateID)
{
	// Prepare image and roi
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

	// Prepare mask
	int iSpan = oTempImage.iSpan;
	int iWidth = oTempImage.iWidth;
	int iHeight = oTempImage.iHeight;
	int iExpansion = CalculateRingHalfWidth(pFid, pTemplateImg->PixelSizeX());
	unsigned char* dilateBuf = new unsigned char[iSpan*iHeight];
	unsigned char* erodeBuf = new unsigned char[iSpan*iHeight];

	for(int i = 0; i < iSpan*iHeight; i++)
	{
		if(oTempImage.pdData[i]>=128) // remove the effect of anti-alias
		{
			dilateBuf[i] = 255;
			erodeBuf[i] =  255;
		}
		else
		{
			dilateBuf[i] = 0;
			erodeBuf[i] =  0;
		}
	}

	Morpho_2d(dilateBuf, iSpan,		// buffer and stride
		0, 0, iWidth, iHeight, // Roi
		iExpansion*2+1, iExpansion*2+1,		// Kernael size
		DILATE);									// Type

	Morpho_2d(erodeBuf, iSpan,		// buffer and stride
		0, 0, iWidth, iHeight, // Roi
		iExpansion*2+1, iExpansion*2+1,		// Kernael size
		ERODE);		// Type

	// Create positive template
	NgcTemplateSt templateSt;
	templateSt._ptTemplate = new VsStCTemplate();
	int iFlag = vsCreate2DTemplate(&oTempImage, templateRect, _iDepth, templateSt._ptTemplate);
	if(iFlag<0)
	{
		delete templateSt._ptTemplate;
		*pTemplateID = -1; 
		delete [] dilateBuf;
		delete [] erodeBuf;
		return(false);
	}

	// Mask positive templae
	unsigned char* pDilateLine = dilateBuf + iSpan*templateRect.lYMin + templateRect.lXMin;
	unsigned char* pErodeLine = erodeBuf + iSpan*templateRect.lYMin + templateRect.lXMin;
	int iCount = 0;
	for(int iy = 0; iy < (int)templateRect.Height(); iy++)
	{
		for(int ix = 0; ix < (int)templateRect.Width(); ix++)
		{
			if(pDilateLine[ix] == pErodeLine[ix])
				templateSt._ptTemplate->pbMask[iCount] = 1; //masked
			else
				templateSt._ptTemplate->pbMask[iCount] = 0;

			iCount++;
		}
		pDilateLine += iSpan;
		pErodeLine += iSpan;
	}	
	iFlag = vsMask2DNgcTemplate(templateSt._ptTemplate);

	// Create negative temaplate
	unsigned char* pbBuf = pTemplateImg->GetBuffer();
	oTempImage.pdData = new unsigned char[oTempImage.iHeight*oTempImage.iSpan];
	for(int i=0; i<oTempImage.iHeight*oTempImage.iSpan; i++)
		oTempImage.pdData[i] = (unsigned char)(255 -pbBuf[i]);
	
	templateSt._ptNegTemplate = new VsStCTemplate();
	iFlag = vsCreate2DTemplate(&oTempImage, templateRect, _iDepth, templateSt._ptNegTemplate);
	if(iFlag<0)
	{
		delete templateSt._ptTemplate;
		delete templateSt._ptNegTemplate;
		*pTemplateID = -1; 
		delete [] oTempImage.pdData;
		delete [] dilateBuf;
		delete [] erodeBuf;
		return(false);
	}

	// Mask negative template
	pDilateLine = dilateBuf + iSpan*templateRect.lYMin + templateRect.lXMin;
	pErodeLine = erodeBuf + iSpan*templateRect.lYMin + templateRect.lXMin;
	iCount = 0;
	for(int iy = 0; iy < (int)templateRect.Height(); iy++)
	{
		for(int ix = 0; ix < (int)templateRect.Width(); ix++)
		{
			if(pDilateLine[ix] == pErodeLine[ix])
				templateSt._ptNegTemplate->pbMask[iCount] = 1; //masked
			else
				templateSt._ptNegTemplate->pbMask[iCount] = 0;

			iCount++;
		}
		pDilateLine += iSpan;
		pErodeLine += iSpan;
	}		
	iFlag = vsMask2DNgcTemplate(templateSt._ptNegTemplate);

	// Fill the struct
	templateSt._bValid = true;
	templateSt._iTemplateID = _iCurrentIndex;
	_iCurrentIndex++;
	templateSt._pFeature = pFid;

	// Add to the list
	_ngcTemplateStList.push_back(templateSt);
	*pTemplateID = templateSt._iTemplateID;

	//clean up
	delete [] oTempImage.pdData;
	delete [] dilateBuf;
	delete [] erodeBuf;

	return(true);
}

// Return the template ID for a feature if a template for it already exists
// otherwise, return -1
int CyberNgcFiducialCorrelation::GetNgcTemplateID(Feature* pFeature)
{
	list<NgcTemplateSt>::const_iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(IsSameTypeSize(pFeature, i->_pFeature))
			return(i->_iTemplateID);
	}

	return(-1);
}

// Calculate half width of ring in pixels for template createion for a feature
unsigned int CyberNgcFiducialCorrelation::CalculateRingHalfWidth(Feature* pFid, double dImageResolution)
{
	// Feature bound box size
	Box box = pFid->GetBoundingBox();
	double dSize = box.Width()>box.Height() ? box.Width() : box.Height();
	
	// Half width of ring in pixels
	double dScale = 8;
	unsigned int iValue = (unsigned int) (dSize/dImageResolution/dScale + 0.5);
	if(iValue > 10) iValue = 10;

	return(iValue);
}
