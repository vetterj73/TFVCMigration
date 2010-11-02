#include "VsNgcAlignment.h"
#include "VsEnvironManager.h"

VsNgcAlignment::VsNgcAlignment(NgcAlignParams params)
{
	// Get environment
	_oVsEnv = VsEnvironManager::Instance().GetEnv();

	// Create images
	_alignSt.oImTemplate = vsCreateCamImageFromBuffer(
		_oVsEnv, NULL, 
		params.pcTemplateBuf, params.iTemplateImWidth, 
		params.iTemplateImHeight, params.iTemplateImSpan,
		VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, FALSE);
	if(_alignSt.oImTemplate == NULL)
		return;

	_alignSt.oImSearch = vsCreateCamImageFromBuffer(
		_oVsEnv, NULL, 
		params.pcSearchBuf, params.iSearchImWidth, 
		params.iSearchImHeight, params.iSearchImSpan,
		VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, FALSE);
	if(_alignSt.oImSearch == NULL)
		return;

	_alignSt.bMask = params.bUseMask;
	if(_alignSt.bMask)
	{
		_alignSt.oMaskIm = vsCreateCamImageFromBuffer(
			_oVsEnv, NULL, 
			params.pcTemplateBuf, params.iTemplateImWidth, 
			params.iTemplateImHeight, params.iTemplateImSpan,
			VS_SINGLE_BUFFER, VS_BUFFER_HOST_BYTE, FALSE);
		if(_alignSt.oMaskIm == NULL)
			return;
	}

	// The rectangle for templat and search 
	/** Need check for validation and memory **/
	_alignSt.tRectTemplate.dAngle = 0;
	_alignSt.tRectTemplate.dCenter[0] = (params.iTemplateLeft+params.iTemplateRight)/2.0;
	_alignSt.tRectTemplate.dCenter[1] = (params.iTemplateTop+params.iTemplateBottom)/2.0;
	_alignSt.tRectTemplate.dWidth = params.iTemplateRight - params.iTemplateLeft;
	_alignSt.tRectTemplate.dHeight = params.iTemplateBottom - params.iTemplateTop;

	_alignSt.tRectSearch.dAngle = 0;
	_alignSt.tRectSearch.dCenter[0] = (params.iSearchLeft+params.iSearchRight)/2.0;
	_alignSt.tRectSearch.dCenter[1] = (params.iSearchTop+params.iSearchBottom)/2.0;
	_alignSt.tRectSearch.dWidth = params.iSearchRight - params.iSearchLeft;
	_alignSt.tRectSearch.dHeight = params.iSearchBottom - params.iSearchTop;

	int iMin;
	if(_alignSt.tRectTemplate.dWidth < _alignSt.tRectTemplate.dHeight)
		iMin = (int)_alignSt.tRectTemplate.dWidth;
	else
		iMin = (int)_alignSt.tRectTemplate.dHeight;
	
	// Pyramid deepth
	_alignSt.iDepth = 2;
	if(iMin>100)
		_alignSt.iDepth = 3;
	if(iMin>200)
		_alignSt.iDepth = 4;
	if(iMin>400)
		_alignSt.iDepth = 5;
	if(iMin>800)
		_alignSt.iDepth = 6;
}

VsNgcAlignment::~VsNgcAlignment(void)
{
	vsDispose(_alignSt.oImTemplate);
	vsDispose(_alignSt.oImSearch);
	vsDispose(_alignSt.oMaskIm);
	
	VsEnvironManager::Instance().ReleaseEnv();
}


bool VsNgcAlignment::Align(NgcAlignResults* pResults)
{
	bool bFlag = Align();

	if(bFlag)
	{
		pResults->dMatchPosX = _alignSt.dMatchPosX;
		pResults->dMatchPosY = _alignSt.dMatchPosY;
		pResults->dCoreScore = _alignSt.dCoreScore;
		pResults->dAmbigScore= _alignSt.dAmbigScore;
	}

	return(bFlag);
}

bool VsNgcAlignment::Align()
{
	VsStCTemplate tTemplate;
	VsStCorrelate tCorrelate;
	
	// Create template
	if (vsCreateCTemplate(
			_alignSt.oImTemplate,
			&_alignSt.tRectTemplate,
			VS_PYRAMID_CORRELATION,
			_alignSt.iDepth,
			VS_BUFFER_HOST_BYTE,
			false,
			&tTemplate) != VS_OK)
	{
		if(tTemplate.iResultFlags & VS_CTEMPLATE_UNIFORM_TEMPLATE)
			_alignSt.eResult = UniformTemplate;
		else
			_alignSt.eResult = OtherTempFailure;
			
		//IP_LOGGER.Write(g_DebugMesageFile, "Failed to create Template:Uniform! \n");
		return(false);
	}
	else
	{
		if(tTemplate.ptCPyrInfo[0].dStdDev < _alignSt.iMinTempStdDev)
		{
			_alignSt.eResult = UniformTemplate;
			
			vsDispose(&tTemplate);
			//IP_LOGGER.Write( "Failed to create Template:SDV=%f! \n", tTemplate.ptCPyrInfo[0].dStdDev);
			return(false);
		}
	}

	// Using mask
	if(_alignSt.bMask)
	{	
		int iSize = (int) (tTemplate.tPixelRect.dWidth * tTemplate.tPixelRect.dHeight);
		unsigned char *pbBufMask = new unsigned char[iSize]; 

		// Now read the mask image and create mask
		vsReadRect(_alignSt.oMaskIm, &tTemplate.tPixelRect, 1, TRUE, pbBufMask);
		int iMaskCnt = 0;
		for (int iI = 0; iI < iSize; iI++)
		{
			if (pbBufMask[iI] == 255) 
			{
				tTemplate.pbMask[iI] = 1; 
				++iMaskCnt;
			}
			else
			{	
				tTemplate.pbMask[iI] = 0; 
			}
		}

		tTemplate.yAllowMasking = TRUE;
		if(vsMask2DNgcTemplate(&tTemplate) < 0) // If add mask failed
		{
			_alignSt.eResult = MaskFailure;
			vsDispose(&tTemplate);
			return(false);
		}
	}

	// Create correlation struct
	if (vsCreateCorrelate(
			_alignSt.oImSearch, 
			&_alignSt.tRectSearch, 
            &tTemplate, 
            &tCorrelate) != VS_OK) 
	{
		_alignSt.eResult = CreateCorFailure;
		vsDispose(&tTemplate);			
		return(false);
    }       
	// Set correlation paramter
	tCorrelate.dGainTolerance			= _alignSt.fCorGainTolerance;
	tCorrelate.dLoResMinScore			= _alignSt.fCorLoResMinScore;
    tCorrelate.dHiResMinScore			= _alignSt.fCorHiResMinScore;
    tCorrelate.ySortResultPoints		= TRUE;
    tCorrelate.iMaxResultPoints			= 2;
    tCorrelate.iDepth					= _alignSt.iDepth;
    tCorrelate.iNumResultPoints			= 0;
	tCorrelate.dFlatPeakThreshPercent	= 4.0 /* CORR_AREA_FLAT_PEAK_THRESH_PERCENT */;
    tCorrelate.dTimeout					= 200;  // Maximum time correlation should ever take
    tCorrelate.iFlatPeakRadiusThresh	= 5;

	// Do correction
	if (vsCorrelate(&tTemplate, &tCorrelate) != VS_OK) 
	{
		if(tCorrelate.iResultFlags & VS_CORRELATE_TIMED_OUT) 
		{
			_alignSt.eResult = TimeOut;
			vsDispose(&tTemplate);	
			vsDispose(&tCorrelate);	
			return(false);
        }
		if(tCorrelate.iResultFlags & VS_CORRELATE_FLAT_CORRELATION_PEAK) 
		{
			_alignSt.eResult = FlatPeak;
			vsDispose(&tTemplate);	
			vsDispose(&tCorrelate);	
			return(false);
		}
        if((tCorrelate.iResultFlags & 0xffff) == 0) 
		{
			_alignSt.eResult = NoMatch;
			vsDispose(&tTemplate);	
			vsDispose(&tCorrelate);	
			return(false);
		}
        else 
		{  // All other errors
			_alignSt.eResult = UnknownFailure;
			vsDispose(&tTemplate);	
			vsDispose(&tCorrelate);	
			return(false);
		}
	}
    else 
	{
		if((tCorrelate.iNumResultPoints == 0) ||
			(tCorrelate.ptCPoint[0].dScore < 0.75))
		{
			_alignSt.eResult = NoMatch;
			vsDispose(&tTemplate);	
			vsDispose(&tCorrelate);	
			return(false);
        }
        else 
		{
			// Success! Report the Matching location on the target image
            _alignSt.dMatchPosX = tCorrelate.ptCPoint[0].dLoc[0];
            _alignSt.dMatchPosY = tCorrelate.ptCPoint[0].dLoc[1];
			_alignSt.dCoreScore = tCorrelate.ptCPoint[0].dScore;

			if(tCorrelate.iNumResultPoints > 1)
				_alignSt.dAmbigScore = tCorrelate.ptCPoint[1].dScore/tCorrelate.ptCPoint[0].dScore;
			else
				_alignSt.dAmbigScore = 1;

            _alignSt.eResult = Found;
        }
    }

	vsDispose(&tTemplate);	
	vsDispose(&tCorrelate);	
	return(true);
}
