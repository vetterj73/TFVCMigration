#include "CorrelationPair.h"

#include "regoff.h"
#include "Ngc.h"
#include "Logger.h"
#include "CorrelationParameters.h"
#include "Utilities.h"
#include "OverlapDefines.h"

#pragma region CorrelationResult class

// Constructors
CorrelationResult::CorrelationResult()
{
	Default();
}

CorrelationResult::CorrelationResult(
	double		dRowOffset,
	double		dColOffset,
	double		dCorrCoeff,
	double		dAmbigScore)
{
	RowOffset	= dRowOffset;
	ColOffset	= dColOffset;
	CorrCoeff	= dCorrCoeff;
	AmbigScore	= dAmbigScore;
}

CorrelationResult::CorrelationResult(const CorrelationResult& b)
{
	*this = b;
}

void CorrelationResult::operator=(const CorrelationResult& b)
{
	RowOffset	= b.RowOffset;
	ColOffset	= b.ColOffset;
	CorrCoeff	= b.CorrCoeff;
	AmbigScore	= b.AmbigScore;
}

// Default values
void CorrelationResult::Default()
{
	RowOffset = 0;
	ColOffset = 0;
	CorrCoeff = 0;
	AmbigScore = -1;
}
#pragma endregion

#pragma region CorrelationPair class

CorrelationPair::CorrelationPair()
{
	_pImg1 = NULL;
	_pImg2 = NULL;
	_pMaskImg = NULL;
	
	_pOverlap = NULL;
	_iIndex  = -1;

	_bUseMask = false;
	_bIsProcessed = false;
	_bGood4Solver = true;

	_bUsedNgc = false;
}

CorrelationPair::CorrelationPair(
	Image* pImg1, 
	Image* pImg2, 
	UIRect roi1, 
	pair<unsigned int, unsigned int> topLeftCorner2, // (column row)
	unsigned int iDecim,
	unsigned int iNgcColSearchExpansion,
	unsigned int iNgcRowSearchExpansion,
	Overlap* pOverlap,
	Image* pMaskImg)
{
	_pImg1 = pImg1;
	_pImg2 = pImg2;
	_pMaskImg = pMaskImg;

	_roi1 = roi1;	
	_roi2.FirstColumn = topLeftCorner2.first;
	_roi2.FirstRow = topLeftCorner2.second;
	_roi2.LastColumn = _roi2.FirstColumn + _roi1.Columns() - 1;
	_roi2.LastRow = _roi2.FirstRow + _roi1.Rows() -1;

	_iDecim = iDecim;
	_iNgcColSearchExpansion = iNgcColSearchExpansion;
	_iNgcRowSearchExpansion = iNgcRowSearchExpansion;

	_pOverlap = pOverlap;
	_iIndex  = -1;

	_bUseMask = false;
	_bIsProcessed = false;
	_bGood4Solver = true;

	_bUsedNgc = false;
}

// Copy for the same parent overlap
// Otherwise, the overlap point need be overrided
CorrelationPair::CorrelationPair(const CorrelationPair& b)
{
	*this = b;
}

// Copy for the same parent overlap
// Otherwise, the overlap point need be overrided
void CorrelationPair::operator=(const CorrelationPair& b)
{
	_pImg1 = b._pImg1;
	_pImg2 = b._pImg2;
	_pMaskImg = b._pMaskImg;

	_roi1 = b._roi1;
	_roi2 = b._roi2;

	_iDecim = b._iDecim;
	_iNgcColSearchExpansion = b._iNgcColSearchExpansion;
	_iNgcRowSearchExpansion = b._iNgcRowSearchExpansion;

	//** Warning, two objects point to the same parents  
	_pOverlap = b._pOverlap;

	_iIndex = b._iIndex;

	_bUseMask = b._bUseMask;
	_bIsProcessed = b._bIsProcessed;
	_bGood4Solver = b._bGood4Solver;

	_bUsedNgc = b._bUsedNgc;
	
	_result = b._result;
}


void CorrelationPair::SetCorrlelationResult(CorrelationResult result)
{
	_result = result;

	_bIsProcessed = true;
}

// Return true if result is available 
bool CorrelationPair::GetCorrelationResult(CorrelationResult* pResult) const
{
	if(!_bIsProcessed)
		return(false);

	*pResult = _result;

	return(true);
}

CorrelationResult CorrelationPair::GetCorrelationResult() const
{
	return(_result);
}

OverlapType CorrelationPair::GetOverlapType()
{
	if(_pOverlap == NULL)
		return NULL_OVERLAP;
	else
		return _pOverlap->GetOverlapType();
}

// Reset to the satus before doing alignment
bool CorrelationPair::Reset()
{
	_bUseMask = false;
	_bIsProcessed = false;
	_bGood4Solver = true;

	_bUsedNgc = false;

	_result.Default();

	return(true);
}

// Do the alignment
// Return true if it is processed
bool CorrelationPair::DoAlignment(bool bApplyCorrSizeUpLimit, bool* pbCorrSizeReduced)
{
	bool bSRC = true;
	_bUsedNgc = false;

	// Check the mask area to decide whehter need to use NGC
	bool bUseMask = _bUseMask && _pMaskImg != NULL && _pMaskImg->GetBuffer() != NULL;
	if(bUseMask)
	{
		// Count pixels for mask
		unsigned int iCount = 0;
		unsigned char* pLineBuf = _pMaskImg->GetBuffer() + 
			_roi1.FirstRow*_pMaskImg->PixelRowStride();
		for(unsigned int iy=_roi1.FirstRow; iy<=_roi1.LastRow; iy++)
		{
			for(unsigned int ix=_roi1.FirstColumn; ix<=_roi1.LastColumn; ix++)
			{
				if(pLineBuf[ix] > 0 )
					iCount++;
			}

			pLineBuf += _pMaskImg->PixelRowStride();
		}

		// Mask area ratio
		double dRate = (double)iCount/(double)_roi1.Rows()/(double)_roi1.Columns();

		// If ratio > Threshold, use NGC
		if( dRate > CorrelationParametersInst.dMaskAreaRatioTh)
			bSRC = false;

		// If ratio is too big, ignore
		if( dRate > 0.95)
			return(false);
	}

	if(bSRC)
	{	// use SRC/regoff
		if(_roi1.Columns()<CorrelationParametersInst.iCorrPairMinRoiSize || _roi1.Rows()<CorrelationParametersInst.iCorrPairMinRoiSize)
		{
			return(false);
		}
		
		if(!SqRtCorrelation(bApplyCorrSizeUpLimit, pbCorrSizeReduced))
			return(false);
	}
	else	// Use Ngc
	{	
		// Mask sure ROI size is bigger enough for search
		if(_roi1.Columns() < 2*_iNgcColSearchExpansion+CorrelationParametersInst.iCorrPairMinRoiSize ||
			_roi1.Rows() < 2*_iNgcRowSearchExpansion+CorrelationParametersInst.iCorrPairMinRoiSize)
			return(false);

		_bUsedNgc = true;

		if(!NGCCorrelation(bApplyCorrSizeUpLimit, pbCorrSizeReduced))
			return(false);
	}
	
	_bIsProcessed = true;
	return(true);
}


// Do the square root correlation and report result
bool CorrelationPair::SqRtCorrelation(bool bApplyCorrSizeUpLimit, bool* pbCorrSizeReduced)
{	
	unsigned int nrows = Rows();
	unsigned int ncols = Columns();

	unsigned int iFirstCol1 = _roi1.FirstColumn;
	unsigned int iFirstRow1 = _roi1.FirstRow;
	unsigned int iFirstCol2 = _roi2.FirstColumn;
	unsigned int iFirstRow2 = _roi2.FirstRow;

	// For Bayer image
	// Create luminance buffer
	unsigned char* pLumBuf1 = NULL;
	unsigned char* pLumBuf2 = NULL;

	if(bApplyCorrSizeUpLimit)
	{
		*pbCorrSizeReduced = false; 

		// Adjust Rows if it is necessary
		if(nrows > CorrelationParametersInst.iCorrMaxRowsToUse && ncols >= CorrelationParametersInst.iCorrMaxColsToUse/2) 
		{
			iFirstRow1 += (nrows - CorrelationParametersInst.iCorrMaxRowsToUse)/2;
			iFirstRow2 += (nrows - CorrelationParametersInst.iCorrMaxRowsToUse)/2;
			nrows = CorrelationParametersInst.iCorrMaxRowsToUse;

			*pbCorrSizeReduced = true;
		}

		// Adjust Cols if it is necessary
		if(ncols > CorrelationParametersInst.iCorrMaxColsToUse && nrows >= CorrelationParametersInst.iCorrMaxRowsToUse/2) 
		{
			iFirstCol1 += (ncols - CorrelationParametersInst.iCorrMaxColsToUse)/2;
			iFirstCol2 += (ncols - CorrelationParametersInst.iCorrMaxColsToUse)/2;
			ncols = CorrelationParametersInst.iCorrMaxColsToUse;
			
			*pbCorrSizeReduced = true;
		}
	}

	/*
	   Note: Due to the fact that a 2-D FFT is taken of the
	   the decimated image, there are restrictions on the
	   permissible values of nrows and ncols.  They must
	   each be factorizable as decim*2^p * 3^q * 5^r.
	   Furthermore, performance is poor if the image
	   dimensions are less than 2*HOOD*decim.  A practical
	   option for images of unsuitable dimensions is to
	   register based on the largest feasible subsection of
	   the image.  Subroutine RegoffLength() is provided to
	   assist in computing suitable dimensions.
	*/
	nrows = RegoffLength(nrows, _iDecim);
	ncols = RegoffLength(ncols, _iDecim);

	// Pointer to first image
	unsigned char* first_image_buffer =
		_pImg1->GetBuffer(iFirstCol1, iFirstRow1);

	// Pointer to second image
	unsigned char* second_image_buffer =
		_pImg2->GetBuffer(iFirstCol2, iFirstRow2);

	int RowStrideA(_pImg1->PixelRowStride());
	int RowStrideB(_pImg2->PixelRowStride());

	/*
		Pointer to space large enough to contain complexf
		array of size at least ncols*nrows/decimx.  The
		array is filled with the correlogram in the .r member
		of each element.  If null pointers are passed for
		this argument, a local array is allocated and freed.
	*/
	complexf *z(0);			


	REGLIM  *lims(0);   /* Limits of search range for registration offset.  Use
						   if there is _a priori_ knowledge about the offset.
						   If a null pointer is passed, the search range
						   defaults to the entire range of values

							  x = [-ncols/2, ncols/2>
							  y = [-nrows/2, nrows/2>

						   Excessively large values are clipped to the above
						   range; inconsistent values result in an error return.
						*/

	int      job(0);   /* 1 = histogram equalize images, 0 = no EQ */
	
	// Enable negative corrlation 
	job |= REGOFF_ABS; 
	//job |= REGOFF_HPNORM;

	int      histclip(1);   /* Histogram clipping factor; clips peaks to prevent
						   noise in large flat regions from being excessively
						   amplified.  Use histclip=1 for no clipping;
						   histclip>1 for clipping.  Recommended value = 32 */

	int      dump(0);       /* Dump intermediate images to TGA files
						   (useful for debugging):

						   ZR.TGA and ZI.TGA are decimated (and possibly
							  histogram-equalized images that are input to the
							  correlation routine.

						   PCORR.TGA is the correlogram.

						   HOLE.TGA is the correlogram, excluding the vicinity
							  of the peak. */


      /* Address of pointer to error message string.  Display
						   to obtain verbal description of error return. */

	char myChar[512];
	char** myCharPtr = (char**)(&myChar);

	CorrelationResult result;

	int iFlag = regoff(	ncols, nrows, first_image_buffer, second_image_buffer, 
			RowStrideA, RowStrideB, z, _iDecim, _iDecim, lims, job, 
			histclip, dump, &result.ColOffset, &result.RowOffset,
			&result.CorrCoeff, &result.AmbigScore, myCharPtr/*&error_msg*/);

	if(iFlag!=0) return(false);

	SetCorrlelationResult(result);

	return(true);
}

// Calculate correlatin by using NGC
bool CorrelationPair::NGCCorrelation(bool bApplyCorrSizeUpLimit, bool* pbCorrSizeReduced)
{
	unsigned int nrows = Rows();
	unsigned int ncols = Columns();

	unsigned int iFirstCol1 = _roi1.FirstColumn;
	unsigned int iFirstRow1 = _roi1.FirstRow;
	unsigned int iFirstCol2 = _roi2.FirstColumn;
	unsigned int iFirstRow2 = _roi2.FirstRow;

	unsigned int iLastCol1 = _roi1.LastColumn;
	unsigned int iLastRow1 = _roi1.LastRow;
	unsigned int iLastCol2 = _roi2.LastColumn;
	unsigned int iLastRow2 = _roi2.LastRow;

	if(bApplyCorrSizeUpLimit)
	{
		*pbCorrSizeReduced = false; 

		// Adjust Rows if it is necessary
		if(nrows > CorrelationParametersInst.iCorrMaxRowsToUse) 
		{
			iFirstRow1 += (nrows - CorrelationParametersInst.iCorrMaxRowsToUse)/2;
			iFirstRow2 += (nrows - CorrelationParametersInst.iCorrMaxRowsToUse)/2;
			nrows = CorrelationParametersInst.iCorrMaxRowsToUse;
			iLastRow1 = iFirstRow1 + nrows - 1;
			iLastRow2 = iFirstRow2 + nrows - 1;

			*pbCorrSizeReduced = true; 
		}

		// Adjust Cols if it is necessary
		if(ncols > CorrelationParametersInst.iCorrMaxColsToUse) 
		{
			iFirstCol1 += (ncols - CorrelationParametersInst.iCorrMaxColsToUse)/2;
			iFirstCol2 += (ncols - CorrelationParametersInst.iCorrMaxColsToUse)/2;
			ncols = CorrelationParametersInst.iCorrMaxColsToUse;
			iLastCol1 = iFirstCol1 + ncols - 1;
			iLastCol2 = iFirstCol2 + ncols - 1;

			*pbCorrSizeReduced = true;
		}
	}

	UIRect tempRect;
	tempRect.FirstColumn = iFirstCol1 + _iNgcColSearchExpansion;
	tempRect.LastColumn = iLastCol1 - _iNgcColSearchExpansion;
	tempRect.FirstRow = iFirstRow1 + _iNgcRowSearchExpansion;
	tempRect.LastRow = iLastRow1 - _iNgcRowSearchExpansion;

	if(!tempRect.IsValid() || 
		tempRect.Columns() < CorrelationParametersInst.iCorrPairMinRoiSize ||
		tempRect.Rows() < CorrelationParametersInst.iCorrPairMinRoiSize)
		return(false);

	UIRect searchRect;
	searchRect.FirstColumn = iFirstCol2;
	searchRect.LastColumn = iLastCol2;
	searchRect.FirstRow = iFirstRow2;
	searchRect.LastRow = iLastRow2;

	int iFlag = MaskedNgc(tempRect, searchRect);
	if(iFlag>=0)
		return(true);
	else
		return(false);
}

// Cyber Ngc correlation with mask
int CorrelationPair::MaskedNgc(UIRect tempRoi, UIRect searchRoi)
{
	// Create template
	SvImage oTempImage;
	oTempImage.pdData = _pImg1->GetBuffer();
	oTempImage.iWidth = _pImg1->Columns();
	oTempImage.iHeight = _pImg1->Rows();
	oTempImage.iSpan = _pImg1->PixelRowStride();
	
	VsStRect templateRect;
	templateRect.lXMin = tempRoi.FirstColumn;
	templateRect.lXMax = tempRoi.LastColumn;
	templateRect.lYMin = tempRoi.FirstRow;
	templateRect.lYMax = tempRoi.LastRow;
	
	// Calculate depth based on template size
		// In width
	int iW = templateRect.Width();
	int iWCount = 0;
	while(iW>2)
	{
		iW /= 2;
		iWCount++;
	}
		// In height
	int iH = templateRect.Height();
	int iHCount = 0;
	while(iH>2)
	{
		iH /= 2;
		iHCount++;
	}
		// Pick the smaller on with some offset
	int iDepth = iWCount<iHCount ? iWCount : iHCount;
	iDepth -= 2;
	if(iDepth<1) iDepth = 1; // The minimum depth is 1
	
	VsStCTemplate tTemplate;
	int iFlag = vsCreate2DTemplate(&oTempImage, templateRect, iDepth, &tTemplate);
	if(iFlag<0) 
		return(-1);	

	// Add mask
	unsigned char* pMaskLine = NULL;
	pMaskLine = (unsigned char*)_pMaskImg->GetBuffer() + _pMaskImg->PixelRowStride()  *templateRect.lYMin + templateRect.lXMin;
	int iCount = 0;
	for(int iy = 0; iy < (int)templateRect.Height(); iy++)
	{
		for(int ix = 0; ix < (int)templateRect.Width(); ix++)
		{
			if(pMaskLine[ix]>0)
				tTemplate.pbMask[iCount] = 1; //masked
			else
				tTemplate.pbMask[iCount] = 0;

			iCount++;
		}
		pMaskLine += _pMaskImg->PixelRowStride();
	}	
	iFlag = vsMask2DNgcTemplate(&tTemplate);
	if(iFlag < 0)
	{
		vsDispose2DTemplate(&tTemplate);
		return(-2);
	}
	
	// Search
	SvImage oSearchImage;
	oSearchImage.pdData = _pImg2->GetBuffer();
	oSearchImage.iWidth = _pImg2->Columns();
	oSearchImage.iHeight = _pImg2->Rows();
	oSearchImage.iSpan = _pImg2->PixelRowStride();
    
	VsStRect searchRect;
	searchRect.lXMin = searchRoi.FirstColumn;
	searchRect.lXMax = searchRoi.LastColumn;
	searchRect.lYMin = searchRoi.FirstRow;
	searchRect.lYMax = searchRoi.LastRow;
	
	// Set correlation paramter	
	VsStCorrelate tCorrelate;
	tCorrelate.dGainTolerance			= 0.5;	// This is ridiciously high, but seems working in this way
	tCorrelate.dLoResMinScore			= 0.25;	// Intentionally low these two value for ambiguous check
    tCorrelate.dHiResMinScore			= 0.25;
    tCorrelate.iMaxResultPoints			= 2;
	// Flat peak check
	//tCorrelate.dFlatPeakThreshPercent	= 4.0 /* CORR_AREA_FLAT_PEAK_THRESH_PERCENT */;
    //tCorrelate.iFlatPeakRadiusThresh	= 5;

	iFlag =vs2DCorrelate(
		&tTemplate, &oSearchImage, 
		searchRect, iDepth, &tCorrelate);
	if(iFlag < 0) // Error or no match
	{
		vsDispose2DTemplate(&tTemplate);	
		vsDispose2DCorrelate(&tCorrelate);	
		return(0);
	}
	if(tCorrelate.iNumResultPoints == 0) // No Match 
	{
		vsDispose2DTemplate(&tTemplate);	
		vsDispose2DCorrelate(&tCorrelate);
		return(0);
	}
	if(tCorrelate.ptCPoint[0].dScore < 0.5) // Match is too low
	{
		vsDispose2DTemplate(&tTemplate);	
		vsDispose2DCorrelate(&tCorrelate);
		return(0);
	}

	// Get results
	_result.ColOffset = tCorrelate.ptCPoint[0].dLoc[0] - (_roi2.FirstColumn+_roi2.LastColumn)/2.0; 
	_result.RowOffset = tCorrelate.ptCPoint[0].dLoc[1] - (_roi2.FirstRow+_roi2.LastRow)/2.0;
	_result.CorrCoeff = tCorrelate.ptCPoint[0].dScore;
	if(tCorrelate.iNumResultPoints >=2)
		_result.AmbigScore= fabs(tCorrelate.ptCPoint[1].dScore/tCorrelate.ptCPoint[0].dScore);
	else
		_result.AmbigScore = 0;

	// Clean up
	vsDispose2DTemplate(&tTemplate);	
	vsDispose2DCorrelate(&tCorrelate);		

	return(1);
}

// Chop correlation pair into a list of smaller pairs/blocks
// iNumBlockX and iNumBlockY: Numbers of smaller blocks in X(columns) and y(rows) direction
// iBlockWidth and iBlockHeight; Size of block in pixel (columns* rows)
// pOutPairlist: output, it will be clean up at the begginning, the list of smaller pairs/blocks
bool CorrelationPair::ChopCorrPair(
	unsigned int iNumBlockX, 
	unsigned int iNumBlockY, 
	unsigned int iBlockWidth, 
	unsigned int iBlockHeight,
	unsigned int iBlockDecim,
	unsigned int iBlockColSearchExpansion,
	unsigned int iBlockRowSearchExpansion,
	list<CorrelationPair>* pOutPairList) const
{
	// Validation check
	if(iNumBlockX<1 || iNumBlockY<1) return(false);	
	
	// Clean up 
	pOutPairList->clear();
	
	// Re_adjust of bloc kwidth and height for protection
	if(iBlockWidth > _roi1.Columns()-(iNumBlockX-1))
		iBlockWidth = _roi1.Columns()-(iNumBlockX-1);

	if(iBlockHeight > _roi1.Rows()-(iNumBlockY-1))
		iBlockHeight = _roi1.Rows()-(iNumBlockY-1);

	// Calculate pitches
	int iPitchX = 0;
	if(iNumBlockX!=1)
		iPitchX = (_roi1.Columns()-iBlockWidth)/(iNumBlockX-1);

	int iPitchY = 0;
	if(iNumBlockY!=1)
		iPitchY = (_roi1.Rows()-iBlockHeight)/(iNumBlockY-1);

	// Calculate blocks
	UIRect roi1, roi2;
	for(unsigned int iy=0; iy<iNumBlockY; iy++)
	{
		// Calculate first and last rows
		unsigned  int iOffsetY; 
		if(iPitchY == 0)	
		{	// Center it if there is only one row
			iOffsetY = (_roi1.Rows() - iBlockHeight)/2;
		}
		else 
			iOffsetY = iy * iPitchY;

		roi1.FirstRow = _roi1.FirstRow + iOffsetY;
		roi1.LastRow = roi1.FirstRow + iBlockHeight - 1;

		roi2.FirstRow = _roi2.FirstRow + iOffsetY;
		roi2.LastRow = roi2.FirstRow + iBlockHeight - 1;

		if(roi1.FirstRow < 0 || 
			roi1.LastRow > _pImg1->Rows()-1 ||
			roi2.FirstRow < 0 || 
			roi2.LastRow > _pImg2->Rows()-1)
		{
			LOG.FireLogEntry(LogTypeError, "CorrelationPair::ChopCorrPair(): ROI is invalid");
			return(false);
		}

		//Calcualte first and last columns
		for(unsigned int ix=0; ix<iNumBlockX; ix++)
		{
			unsigned  int iOffsetX; 
			if(iPitchX == 0)	
			{	// Center it if there is only one column
				iOffsetX = (_roi1.Columns() - iBlockWidth)/2;
			}
			else 
				iOffsetX = ix * iPitchX;

			roi1.FirstColumn = _roi1.FirstColumn + iOffsetX;
			roi1.LastColumn = roi1.FirstColumn + iBlockWidth - 1;

			roi2.FirstColumn = _roi2.FirstColumn + iOffsetX;
			roi2.LastColumn = roi2.FirstColumn + iBlockWidth - 1;

			if(roi1.FirstColumn < 0 || 
				roi1.LastColumn > _pImg1->Columns()-1 ||
				roi2.FirstColumn < 0 || 
				roi2.LastColumn > _pImg2->Columns()-1)
			{
				LOG.FireLogEntry(LogTypeError, "CorrelationPair::ChopCorrPair(): ROI is invalid");
				return(false);
			}

			CorrelationPair corrPair(
				_pImg1,
				_pImg2,
				roi1,
				pair<unsigned int, unsigned int>(roi2.FirstColumn, roi2.FirstRow), // (column row)
				iBlockDecim,
				iBlockColSearchExpansion,
				iBlockRowSearchExpansion,
				_pOverlap,
				_pMaskImg);

			// Set the fine overlap index
			corrPair.SetIndex(iy*10+ix);

			pOutPairList->push_back(corrPair);
		}
	}

	return(true);
}

// Create a copy of correlation pair with ROI are adjust from the correaltion result
// pPair: output, the created correlation pair if success
bool CorrelationPair::AdjustRoiBaseOnResult(CorrelationPair* pPair) const
{
	// Validation check
	if(!_bIsProcessed)
		return(false);

	// Adjustment offsets
	int col_offset = (int)(_result.ColOffset + 0.5);	
	int row_offset = (int)(_result.RowOffset + 0.5);

	// validation check
	if(_roi1.Columns() < abs(col_offset)+CorrelationParametersInst.iCorrPairMinRoiSize ||
		_roi1.Rows() < abs(col_offset)+CorrelationParametersInst.iCorrPairMinRoiSize)
		return(false);

	*pPair = *this;

	// Adjust for rois
	if( col_offset>0 )
	{
		pPair->_roi2.FirstColumn += col_offset;
		pPair->_roi1.LastColumn -= col_offset;
	}
	else if( col_offset<0 )
	{
		pPair->_roi2.LastColumn += col_offset;
		pPair->_roi1.FirstColumn -= col_offset;
	}

	// Adjust for colums 
	if( row_offset>0 )
	{
		pPair->_roi2.FirstRow += row_offset;
		pPair->_roi1.LastRow -= row_offset;
	}
	else if( row_offset<0 )
	{
		pPair->_roi2.LastRow += row_offset;
		pPair->_roi1.FirstRow -= row_offset;
	}

	return(true);
}

void CorrelationPair::NorminalCenterInWorld(double* pdx, double* pdy)
{
	_pImg1->ImageToWorld(_roi1.RowCenter(), _roi1.ColumnCenter(), pdx, pdy); 
}

// For Debug
void CorrelationPair::DumpImg(string sFileName) const
{
	// for debug
	//if(!_bUsedNgc) return;

	unsigned char* pcBuf1 = _pImg1->GetBuffer() 
		+ _pImg1->PixelRowStride()*_roi1.FirstRow
		+ _roi1.FirstColumn;

	unsigned char* pcBuf2 = _pImg2->GetBuffer() 
		+ _pImg2->PixelRowStride()*_roi2.FirstRow
		+ _roi2.FirstColumn;
	
	Bitmap* rgb; 
	if(!_bUsedNgc)
	{
		rgb = Bitmap::New2ChannelBitmap( 
			_roi1.Rows(), 
			_roi1.Columns(),
			pcBuf1, 
			pcBuf2,
			_pImg1->PixelRowStride(),
			_pImg2->PixelRowStride() );
	}
	else
	{
		unsigned char* pcBuf3 = _pMaskImg->GetBuffer()
			+ _pMaskImg->PixelRowStride()*_roi1.FirstRow
			+ _roi1.FirstColumn;

		rgb = Bitmap::New3ChannelBitmap(
			_roi1.Rows(), 
			_roi1.Columns(),
			pcBuf1, 
			pcBuf2,
			pcBuf3,
			_pImg1->PixelRowStride(),
			_pImg2->PixelRowStride(),
			_pMaskImg->PixelRowStride());
	}

	rgb->write(sFileName);

	delete rgb;
}

bool CorrelationPair::DumpImgWithResult(string sFileName) const
{
	// For debug	
	//if(!_bUsedNgc) return(false);
	
	if(!_bIsProcessed) return(false);	

// For first channel
	unsigned char* pcBuf1 = _pImg1->GetBuffer() 
		+ _pImg1->PixelRowStride()*_roi1.FirstRow
		+ _roi1.FirstColumn;	

	int iWidth = _roi1.Columns();
	int iHeight = _roi1.Rows();
	int ix, iy;

// For second channel
	unsigned char* pcBuf2 = _pImg2->GetBuffer() 
		+ _pImg2->PixelRowStride()*_roi2.FirstRow
		+ _roi2.FirstColumn;

	int iSpan = _pImg2->PixelRowStride();
	unsigned char* pcTempBuf2 = new Byte[iWidth*iHeight];
	::memset(pcTempBuf2, 0, iWidth*iHeight);

	int iOffsetX = (int)_result.ColOffset;
	int iOffsetY = (int)_result.RowOffset;

	// Move second ROI image patch to match first ROI image patch
	int iLocX, iLocY;
	for(iy=0; iy<iHeight; iy++)
	{
		for(ix=0; ix<iWidth; ix++)
		{
			iLocX = ix+iOffsetX;
			iLocY = iy+iOffsetY;

			// out of second ROI 
			if(iLocX<0 || iLocY<0 || iLocX>=iWidth || iLocY>=iHeight)
				continue;

			pcTempBuf2[iy*iWidth+ ix] =pcBuf2[iSpan*iLocY+iLocX]; 
		}
	}

// Create bmp
	Bitmap* rgb;
	if(!_bUsedNgc)
	{
		rgb = Bitmap::New2ChannelBitmap( 
			_roi1.Rows(), 
			_roi1.Columns(),
			pcBuf1, 
			pcTempBuf2,
			_pImg1->PixelRowStride(),
			iWidth );
	}
	else
	{
		unsigned char* pcBuf3 = _pMaskImg->GetBuffer()
			+ _pMaskImg->PixelRowStride()*_roi1.FirstRow
			+ _roi1.FirstColumn;

		rgb = Bitmap::New3ChannelBitmap(
			_roi1.Rows(), 
			_roi1.Columns(),
			pcBuf1, 
			pcTempBuf2,
			pcBuf3,
			_pImg1->PixelRowStride(),
			iWidth,
			_pMaskImg->PixelRowStride());
	}


	rgb->write(sFileName);

	delete rgb;
	delete [] pcTempBuf2;

	return(true);
}

#pragma endregion