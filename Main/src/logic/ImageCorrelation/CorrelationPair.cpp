#include "CorrelationPair.h"

#include "SquareRootCorrelation.h"
#include "VsNgcWrapper.h"
#include "Logger.h"
#include "CorrelationParameters.h"


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
	
	_type = NULL_OVERLAP;

	_bIsProcessed=false;
}

CorrelationPair::CorrelationPair(
		Image* pImg1, 
		Image* pImg2, 
		UIRect roi1, 
		pair<unsigned int, unsigned int> topLeftCorner2,
		unsigned int iDecim,
		unsigned int iColSearchExpansion,
		unsigned int iRowSearchExpansion,
		OverlapType type,
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
	_iColSearchExpansion = iColSearchExpansion;
	_iRowSearchExpansion = iRowSearchExpansion;

	_type = type;

	_bIsProcessed = false;
}

CorrelationPair::CorrelationPair(const CorrelationPair& b)
{
	*this = b;
}

void CorrelationPair::operator=(const CorrelationPair& b)
{
	_pImg1 = b._pImg1;
	_pImg2 = b._pImg2;
	_pMaskImg = b._pMaskImg;

	_roi1 = b._roi1;
	_roi2 = b._roi2;

	_iDecim = b._iDecim;
	_iColSearchExpansion = b._iColSearchExpansion;
	_iRowSearchExpansion = b._iRowSearchExpansion;

	_type = b._type;

	_bIsProcessed = b._bIsProcessed;
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

// Reset to the satus before doing alignment
bool CorrelationPair::Reset()
{
	_bIsProcessed = false;
	_result.Default();

	return(true);
}

// Do the alignment
// Return true if it is processed
bool CorrelationPair::DoAlignment()
{
	bool bSRC = true;

	// Check the mask area to decide whehter need to use NGC
	if(_pMaskImg != NULL && _pMaskImg->GetBuffer() != NULL)
	{
		unsigned int iCount = 0;
		unsigned char* pLineBuf = _pMaskImg->GetBuffer() + 
			_roi1.FirstRow*_pImg1->ByteRowStride();
		for(unsigned int iy=_roi1.FirstRow; iy=_roi1.LastRow; iy++)
		{
			for(unsigned int ix=_roi1.FirstColumn; ix<_roi1.LastColumn; ix++)
			{
				if(pLineBuf[ix] > 0 )
					iCount++;
			}

			pLineBuf += _pImg1->ByteRowStride();
		}

		if(iCount*100/_roi1.Rows()/_roi1.Columns() > CorrParams.dMaskAreaRatioTh*100)
			bSRC = false;
	}

	if(bSRC)
	{	// use SRC/regoff
		if(_roi1.Columns()<CorrParams.iCorrPairMinRoiSize || _roi1.Rows()<CorrParams.iCorrPairMinRoiSize)
		{
			return(false);
		}
		SqRtCorrelation(this, _iDecim, true);
	}
	else
	{	
		// Mask sure ROI size is bigger enough for search
		if(_roi1.Columns() < 2*_iColSearchExpansion+CorrParams.iCorrPairMinRoiSize ||
			_roi1.Rows() < 2*_iRowSearchExpansion+CorrParams.iCorrPairMinRoiSize)
			return(false);

		// Use Ngc
		// Ngc input parameter
		NgcParams params;
		params.pcTemplateBuf	= _pImg1->GetBuffer();
		params.iTemplateImWidth = _pImg1->Columns();
		params.iTemplateImHeight= _pImg1->Rows();
		params.iTemplateImSpan	= _pImg1->PixelRowStride();
		params.iTemplateLeft	= _roi1.FirstColumn + _iColSearchExpansion;
		params.iTemplateRight	= _roi1.LastColumn - _iColSearchExpansion;
		params.iTemplateTop		= _roi1.FirstRow + _iRowSearchExpansion;
		params.iTemplateBottom	= _roi1.LastRow - _iRowSearchExpansion;

		params.pcSearchBuf		= _pImg2->GetBuffer();
		params.iSearchImWidth	= _pImg2->Columns();
		params.iSearchImHeight	= _pImg2->Rows();
		params.iSearchImSpan	= _pImg2->PixelRowStride();
		params.iSearchLeft		= _roi2.FirstColumn;
		params.iSearchRight		= _roi2.LastColumn;
		params.iSearchTop		= _roi2.FirstRow;
		params.iSearchBottom	= _roi2.LastRow;

		params.bUseMask			= true;
		params.pcMaskBuf		= _pMaskImg->GetBuffer();

		// Ngc alignment
		NgcResults results;
		VsNgcWrapper ngc;
		ngc.Align(params, &results);

		/** Check this **/
		// Create result
		_result.ColOffset = results.dMatchPosX - (_roi2.FirstColumn+_roi2.LastColumn)/2.0; 
		_result.RowOffset = results.dMatchPosY - (_roi2.FirstRow+_roi2.LastRow)/2.0;
		_result.CorrCoeff = results.dCoreScore;
		_result.AmbigScore= results.dAmbigScore;
	}
	
	_bIsProcessed = true;
	
	return(true);
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
	if(iBlockWidth > _roi1.Columns()-iNumBlockX)
		iBlockWidth = _roi1.Columns()-iNumBlockX;

	if(iBlockHeight > _roi1.Rows()-iNumBlockY)
		iBlockHeight = _roi1.Rows()-iNumBlockY;

	// Calculate pitches
	int iPitchX = 0;
	if(iNumBlockX!=1)
		iPitchX = (_roi1.Columns()-iBlockWidth)/iNumBlockX;

	int iPitchY = 0;
	if(iNumBlockY!=1)
		iPitchY = (_roi1.Rows()-iBlockHeight)/iNumBlockY;

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
		roi1.LastRow = _roi1.FirstRow + iBlockHeight - 1;

		roi2.FirstRow = _roi2.FirstRow + iOffsetY;
		roi2.LastRow = _roi2.FirstRow + iBlockHeight - 1;

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
			roi1.LastColumn = _roi1.FirstColumn + iBlockWidth - 1;

			roi2.FirstColumn = _roi2.FirstColumn + iOffsetX;
			roi2.LastColumn = _roi2.FirstColumn + iBlockWidth - 1;

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
				pair<unsigned int, unsigned int>(roi2.FirstColumn, roi2.FirstRow),
				iBlockDecim,
				iBlockColSearchExpansion,
				iBlockRowSearchExpansion,
				_type,
				_pMaskImg);

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
	if(_roi1.Columns() < abs(col_offset)+CorrParams.iCorrPairMinRoiSize ||
		_roi1.Rows() < abs(col_offset)+CorrParams.iCorrPairMinRoiSize)
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

// For Debug
void CorrelationPair::DumpImg(string sFileName) const
{
	unsigned char* pcBuf1 = _pImg1->GetBuffer() 
		+ _pImg1->PixelRowStride()*_roi1.FirstRow
		+ _roi1.FirstColumn;

	unsigned char* pcBuf2 = _pImg2->GetBuffer() 
		+ _pImg2->PixelRowStride()*_roi2.FirstRow
		+ _roi2.FirstColumn;

	Bitmap* rbg = Bitmap::New2ChannelBitmap( 
		_roi1.Rows(), 
		_roi1.Columns(),
		pcBuf1, 
		pcBuf2,
		_pImg1->PixelRowStride(),
		_pImg2->PixelRowStride() );

	rbg->write(sFileName);

	delete rbg;
}

bool CorrelationPair::DumpImgWithResult(string sFileName) const
{
	if(!_bIsProcessed) return(false);

	unsigned char* pcBuf1 = _pImg1->GetBuffer() 
		+ _pImg1->PixelRowStride()*_roi1.FirstRow
		+ _roi1.FirstColumn;

	unsigned char* pcBuf2 = _pImg2->GetBuffer() 
		+ _pImg2->PixelRowStride()*_roi2.FirstRow
		+ _roi2.FirstColumn;

	int iWidth = _roi1.Columns();
	int iHeight = _roi1.Rows();
	int iSpan = _pImg2->PixelRowStride();
	Byte* pcTempBuf = new Byte[iWidth*iHeight];
	::memset(pcTempBuf, 0, iWidth*iHeight);

	int iOffsetX = (int)_result.ColOffset;
	int iOffsetY = (int)_result.RowOffset;

	// Move second ROI image patch to match first ROI image patch
	int ix, iy, iLocX, iLocY;
	for(iy=0; iy<iHeight; iy++)
	{
		for(ix=0; ix<iWidth; ix++)
		{
			iLocX = ix+iOffsetX;
			iLocY = iy+iOffsetY;

			// out of second ROI 
			if(iLocX<0 || iLocY<0 || iLocX>=iWidth || iLocY>=iHeight)
				continue;

			pcTempBuf[iy*iWidth+ ix] =pcBuf2[iSpan*iLocY+iLocX]; 
		}
	}

	Bitmap* rbg = Bitmap::New2ChannelBitmap( 
		_roi1.Rows(), 
		_roi1.Columns(),
		pcBuf1, 
		pcTempBuf,
		_pImg1->PixelRowStride(),
		iWidth );

	rbg->write(sFileName);

	delete rbg;

	delete [] pcTempBuf;

	return(true);
}

#pragma endregion