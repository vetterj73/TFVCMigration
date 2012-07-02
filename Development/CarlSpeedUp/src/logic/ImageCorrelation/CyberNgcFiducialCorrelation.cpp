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
		_ptPosTemplate = NULL;
		_ptNegTemplate = NULL;
		_bValid = false;
	};

	~NgcTemplateSt()
	{
	};

	Feature* _pFeature;
	VsStCTemplate* _ptPosTemplate;
	VsStCTemplate* _ptNegTemplate;
	unsigned int _iTemplateID;
	bool _bValid;
};


// For singleton pattern
CyberNgcFiducialCorrelation* CyberNgcFiducialCorrelation::pInstance = 0;
CyberNgcFiducialCorrelation& CyberNgcFiducialCorrelation::Instance()
{
	if( pInstance == NULL )
		pInstance = new CyberNgcFiducialCorrelation();

	return *pInstance;
}

CyberNgcFiducialCorrelation::CyberNgcFiducialCorrelation(void)
{
	_iCurrentIndex = 0;

	_iDepth = 3; 

	//_entryMutex = CreateMutex(0, FALSE, "Entry Mutex"); // No initial owner
	_entryMutex = CreateMutex(0, FALSE, NULL); // No initial owner
}


CyberNgcFiducialCorrelation::~CyberNgcFiducialCorrelation(void)
{
	list<NgcTemplateSt>::iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(i->_bValid)
		{
			if(i->_ptPosTemplate!=NULL) 
			{
				vsDispose2DTemplate(i->_ptPosTemplate);
				delete i->_ptPosTemplate;
			}
			if(i->_ptNegTemplate!=NULL)
			{
				vsDispose2DTemplate(i->_ptNegTemplate);
				delete i->_ptNegTemplate;
			}
		}
	}

	_ngcTemplateStList.clear();

	CloseHandle(_entryMutex);
}

// Create Ngc template for a fiducial if it doesn't exist
// pFid: input, fiducial feature
// pTemplateImg and tempRect: input, tempate image and rectangle
// Return template ID for existing or new create ngc template
// Return -1 if failed
int CyberNgcFiducialCorrelation::CreateNgcTemplate(
	Feature* pFid, 
	bool bFidBrighterThanBackground,
	bool bFiducialAllowNegativeMatch,
	const Image* pTemplateImg,  // Always Fiducial is brighter than background in image
	UIRect tempRoi)
{
	// Mutex protection
	WaitForSingleObject(_entryMutex, INFINITE);

	bool bCreatePostiveTemplate = true;
	if(!bFidBrighterThanBackground && !bFiducialAllowNegativeMatch)
		bCreatePostiveTemplate = false;
	bool bCreateNegtiveTemplate = true;
	if(bFidBrighterThanBackground && !bFiducialAllowNegativeMatch)
		bCreateNegtiveTemplate = false;

	// If the template exists
	int iTemplateID = GetNgcTemplateID(pFid, bCreatePostiveTemplate, bCreateNegtiveTemplate);
	if(iTemplateID >= 0) return(iTemplateID);
	
	// Create a new template
	bool bFlag = CreateNgcTemplate(pFid, bCreatePostiveTemplate, bCreateNegtiveTemplate, 
		pTemplateImg, tempRoi, &iTemplateID);
	if(!bFlag)
	{
		ReleaseMutex(_entryMutex);
		return(-1);
	}

	// Mutex protection
	ReleaseMutex(_entryMutex);
	
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
	VsStCTemplate* ptPosTemplate = NULL;
	VsStCTemplate* ptNegTemplate = NULL;
	list<NgcTemplateSt>::iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		if(i->_iTemplateID == iNodeID)
		{
			ptPosTemplate = i->_ptPosTemplate;
			ptNegTemplate = i->_ptNegTemplate;
			break;
		}
	}

	// Validation check
	if(ptPosTemplate == NULL && ptNegTemplate == NULL)
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

	double x1, x2, y1, y2;
	double corr1=0, corr2=0, ambig1=1, ambig2=1;
	bool bSuccess1=false, bSuccess2=false;

	// Try postive template
	int iFlag;
	if(ptPosTemplate != NULL)
	{
		iFlag =vs2DCorrelate(
			ptPosTemplate, &oSearchImage, 
			searchRect, _iDepth, &tCorrelate);
		if(iFlag < 0 || tCorrelate.iNumResultPoints == 0 || fabs(tCorrelate.ptCPoint[0].dScore) < dMinScore) // Error or no match
		{	
			vsDispose2DCorrelate(&tCorrelate);
		}
		else
		{
			bSuccess1= true;

			// Get results
			x1 = tCorrelate.ptCPoint[0].dLoc[0]; 
			y1 = tCorrelate.ptCPoint[0].dLoc[1];
			corr1 = tCorrelate.ptCPoint[0].dScore;
			if(tCorrelate.iNumResultPoints >=2)
				ambig1= fabs(tCorrelate.ptCPoint[1].dScore/tCorrelate.ptCPoint[0].dScore);
			else
				ambig1 = 0;

			vsDispose2DCorrelate(&tCorrelate);
	
			if(ptNegTemplate != NULL) // Negative template is available
			{
				if(corr1 > 0.7  &&ambig1 < 0.5 )	// if it is good enough
				{
					x = x1;
					y = y1;
					correlation = corr1;
					ambig = ambig1;
					return(true);
				}	
			}
		}
	}

	if(ptNegTemplate != NULL)
	{
		// Try negative template
		iFlag =vs2DCorrelate(
			ptNegTemplate, &oSearchImage, 
			searchRect, _iDepth, &tCorrelate);
		if(iFlag < 0 || tCorrelate.iNumResultPoints == 0 || fabs(tCorrelate.ptCPoint[0].dScore) < dMinScore) // Error or no match
		{	
			vsDispose2DCorrelate(&tCorrelate);
		}
		else
		{
			bSuccess2= true;

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
bool CyberNgcFiducialCorrelation::CreateNgcTemplate(
	Feature* pFid,
	bool bCreatePositiveTemplate,
	bool bCreateNegativeTemplate,
	const Image* pTemplateImg, 
	UIRect tempRoi, int* pTemplateID)
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

	Morpho_2d(dilateBuf, iSpan,			// buffer and stride
		0, 0, iWidth, iHeight,			// Roi
		iExpansion*2+1, iExpansion*2+1,	// Kernael size
		DILATE);						// Type

	Morpho_2d(erodeBuf, iSpan,			// buffer and stride
		0, 0, iWidth, iHeight,			// Roi
		iExpansion*2+1, iExpansion*2+1,	// Kernael size
		ERODE);							// Type

	NgcTemplateSt templateSt;
	templateSt._ptPosTemplate = NULL;
	templateSt._ptNegTemplate = NULL;
	
	// Create positive template
	if(bCreatePositiveTemplate)
	{
		templateSt._ptPosTemplate = new VsStCTemplate();
		int iFlag = vsCreate2DTemplate(&oTempImage, templateRect, _iDepth, templateSt._ptPosTemplate);
		if(iFlag<0)
		{
			delete templateSt._ptPosTemplate;
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
					templateSt._ptPosTemplate->pbMask[iCount] = 1; //masked
				else
					templateSt._ptPosTemplate->pbMask[iCount] = 0;

				iCount++;
			}
			pDilateLine += iSpan;
			pErodeLine += iSpan;
		}	
		iFlag = vsMask2DNgcTemplate(templateSt._ptPosTemplate);
	}

	// Create negative temaplate	
	if(bCreateNegativeTemplate)
	{
		unsigned char* pbBuf = pTemplateImg->GetBuffer();
		oTempImage.pdData = new unsigned char[oTempImage.iHeight*oTempImage.iSpan];
		for(unsigned int i=0; i<oTempImage.iHeight*oTempImage.iSpan; i++)
			oTempImage.pdData[i] = (unsigned char)(255 -pbBuf[i]);

		templateSt._ptNegTemplate = new VsStCTemplate();
		int iFlag = vsCreate2DTemplate(&oTempImage, templateRect, _iDepth, templateSt._ptNegTemplate);
		if(iFlag<0)
		{
			delete templateSt._ptPosTemplate;
			delete templateSt._ptNegTemplate;
			*pTemplateID = -1; 
			delete [] oTempImage.pdData;
			delete [] dilateBuf;
			delete [] erodeBuf;
			return(false);
		}

		// Mask negative template
		unsigned char* pDilateLine = dilateBuf + iSpan*templateRect.lYMin + templateRect.lXMin;
		unsigned char* pErodeLine = erodeBuf + iSpan*templateRect.lYMin + templateRect.lXMin;
		int iCount = 0;
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
		
		delete [] oTempImage.pdData;
	}

	// Fill the struct
	templateSt._bValid = true;
	templateSt._iTemplateID = _iCurrentIndex;
	_iCurrentIndex++;
	templateSt._pFeature = pFid;

	// Add to the list
	_ngcTemplateStList.push_back(templateSt);
	*pTemplateID = templateSt._iTemplateID;

	//clean up
	delete [] dilateBuf;
	delete [] erodeBuf;

	return(true);
}

// Return the template ID for a feature if a template for it already exists
// otherwise, return -1
int CyberNgcFiducialCorrelation::GetNgcTemplateID(
	Feature* pFeature,
	bool bHasPositiveTemplate,
	bool bHasNegativeTemplate)
{
	list<NgcTemplateSt>::const_iterator i;
	for(i=_ngcTemplateStList.begin(); i!=_ngcTemplateStList.end(); i++)
	{
		bool bCreatedPosTemp = true;
		if(i->_ptPosTemplate == NULL)
			bCreatedPosTemp = false;

		bool bCreatedNegTemp = true;
		if(i->_ptNegTemplate == NULL)
			bCreatedNegTemp = false;

		if(bHasPositiveTemplate == bCreatedPosTemp &&
			bHasNegativeTemplate == bCreatedNegTemp &&
			IsSameTypeSize(pFeature, i->_pFeature))
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
