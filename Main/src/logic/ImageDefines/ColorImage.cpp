#include "ColorImage.h"

ColorImage::ColorImage(COLORSTYLE colorStyle, bool bChannelStoredSeperate)
	:Image(3)
{
	_colorStyle = colorStyle;
	_bChannelStoredSeperate = bChannelStoredSeperate;
}

ColorImage::~ColorImage(void)
{
}

void ColorImage::SetColorStyle(COLORSTYLE value)
{
	if(_colorStyle == value)
		return;
	
	unsigned char* pLine = _buffer;
	int iChannelStep = PixelRowStride()*_rows;

	for(unsigned int iy=0; iy<_rows; iy++) 
	{
		for(unsigned int ix=0; ix<_columns; ix++)
		{
			// Get address for data
			int iAddress[3];
			for(int i=0; i<3; i++)
			{
				if(_bChannelStoredSeperate)
					iAddress[i] = ix+i*iChannelStep;
				else
					iAddress[i] = ix*3+i;
			}
				
			int iTemp[3];			
			// YCrCb to BGR conversion
			if(_colorStyle == YCrCb && value == BGR)
			{
				iTemp[2] = (int)pLine[iAddress[0]] + ((int)pLine[iAddress[1]]-128)*2;									// R
				iTemp[1] = (int)pLine[iAddress[0]] - ((int)pLine[iAddress[1]]-128) - ((int)pLine[iAddress[2]]-128);	// G
				iTemp[0] = (int)pLine[iAddress[0]] + ((int)pLine[iAddress[2]]-128)*2;	// B
			}
			// BGR to YCrCb conversion
			if(_colorStyle ==  BGR && value == YCrCb)
			{
				iTemp[0] = (pLine[iAddress[2]]>>2) + (pLine[iAddress[1]]>>1) + (pLine[iAddress[0]]>>2);	// Y	
				iTemp[1] = (pLine[iAddress[2]] - iTemp[0])/2+128;										// Cr
				iTemp[2] = (pLine[iAddress[0]] - iTemp[0])/2+128;									// Cb
			}

			// Write back
			for(int i=0; i<3; i++)
			{
				if(iTemp[i]<0) iTemp[i]=0;
				if(iTemp[i]>255) iTemp[i]=255;
				pLine[iAddress[i]] = iTemp[i];
			}
		}
		// Next line in the output buffer
		if(_bChannelStoredSeperate)
			pLine += PixelRowStride();
		else
			pLine += ByteRowStride();		
	}

	_colorStyle = value;
}

void ColorImage::SetChannelStoreSeperated(bool bValue)
{
	if(_bChannelStoredSeperate == bValue)
		return;

	unsigned char* pTempBuf = new unsigned char[BufferSizeInBytes()];
	int iChannelStep = PixelRowStride()*_rows;
	unsigned char* pInLine = _buffer;
	unsigned char* pOutLine = pTempBuf;

	// Seperate channel to combined channel		
	for(unsigned int iy=0; iy<_rows; iy++) 
	{
		for(unsigned int ix=0; ix<_columns; ix++)
		{
			for(int i=0; i<3; i++)
			{
				if(_bChannelStoredSeperate)
					pOutLine[ix*3+i] = pInLine[ix+i*iChannelStep];
				else
					pOutLine[ix+i*iChannelStep] = pInLine[ix*3+i];
			}
		}
		// Next line in the output buffer
		if(_bChannelStoredSeperate)
		{			
			pInLine += PixelRowStride();
			pOutLine += ByteRowStride();
		}
		else
		{
			pInLine += ByteRowStride();
			pOutLine += PixelRowStride();
		}
	}

	::memcpy(_buffer, pTempBuf, BufferSizeInBytes());
	_bChannelStoredSeperate = bValue;

	delete [] pTempBuf;
}

bool  ColorImage::DemosaicFrom(const Image* bayerImg, BayerType type)
{
	int iRows = bayerImg->Rows();
	int iCols = bayerImg->Columns();
	int iBayerSpan = bayerImg->PixelRowStride();

	// If size in pixel is not the same
	if(_rows != iRows || iCols != iCols)
	{
		_rows = iRows;
		_columns = iCols;
		_pixelRowStride = iCols;

		DeleteBufferIfOwner();
		_buffer = new unsigned char[BufferSizeInBytes()];
		_IOwnMyOwnBuffer = true;
	}

	int iOutSpan =  _pixelRowStride;
	if(_colorStyle!=YONLY && _bChannelStoredSeperate)
		iOutSpan = ByteRowStride();
	
	BayerLum(                  
		iCols,        
		iRows,
		bayerImg->GetBuffer(),      
		bayerImg->PixelRowStride(),       
		type,          
		_buffer,         
		iOutSpan,       
		_colorStyle,				
		_bChannelStoredSeperate);

	_thisToWorld = bayerImg->GetTransform(); 
	_nominalTrans = bayerImg->GetNominalTransform();

	return(true);
}

bool ColorImage::DemosiacFrom(unsigned char* pBayerBuf, int iCols, int iRows, int iSpan, BayerType type)
{
	// If size in pixel is not the same
	if(_rows != iRows || _columns != iCols)
	{
		_rows = iRows;
		_columns = iCols;
		_pixelRowStride = iCols;

		DeleteBufferIfOwner();
		_buffer = new unsigned char[BufferSizeInBytes()];
		_IOwnMyOwnBuffer = true;
	}

	int iOutSpan =  _pixelRowStride;
	if(_colorStyle!=YONLY && !_bChannelStoredSeperate)
		iOutSpan = ByteRowStride();
	
	BayerLum(                  
		iCols,        
		iRows,
		pBayerBuf,      
		iSpan,       
		type,          
		_buffer,         
		iOutSpan,       
		_colorStyle,				
		_bChannelStoredSeperate);

	return(true);
}


bool  ColorImage::ColorMorphFrom(const ColorImage* pImgIn, UIRect roi)
{
		/*

	[x]			[Row_in]		[Row_out]
	[y] ~= TIn*	[Col_in] ~=Tout*[Col_out]
	[1]			[1     ]		[	   1]

	[Row_in]					[Row_out]	[A00 A01 A02]	[Row_out]
	[Col_in] ~= Inv(TIn)*TOut*	[Col_out] =	[A10 A11 A12] *	[Col_out]
	[1	   ]					[1		]	[A20 A21 1	]	[1		]

	[Col_in]	[A11 A10 A12]	[Col_out]
	[Row_in] ~=	[A01 A00 A02] *	[Row_out]
	[1	   ]	[A21 A20 1	]	[1		]
	*/

	// Validation check (only for 8-bit image)
	if(_bytesPerPixel != 3) return(false);
	
	// Create tansform matrix from (Col_out, Row_out) to (Col_in, Row_in)
	ImgTransform tIn_inv = pImgIn->GetTransform().Inverse();
	ImgTransform t = tIn_inv * _thisToWorld;
	double dTemp[3][3];
	t.GetMatrix(dTemp);
	double dT[3][3];

	dT[0][0] = dTemp[1][1];
	dT[0][1] = dTemp[1][0];
	dT[0][2] = dTemp[1][2];
	
	dT[1][0] = dTemp[0][1];
	dT[1][1] = dTemp[0][0];
	dT[1][2] = dTemp[0][2];

	dT[2][0] = dTemp[2][1];
	dT[2][1] = dTemp[2][0];
	dT[2][2] = dTemp[2][2];

	// Image morph: YCrCb in sperated channels to BGR in combined channels
	ColorImageMorph(
		pImgIn->GetBuffer(), pImgIn->PixelRowStride(),
		pImgIn->Columns(), pImgIn->Rows(),
		_buffer, ByteRowStride(),
		roi.FirstColumn, roi.FirstRow,
		roi.Columns(), roi.Rows(),
		dT);

	return(true);
}

bool  ColorImage::ColorMorphFromWithHeight(
	ColorImage* pImgIn, 
	UIRect roi,
	const Image* pHeightImg, 
	double dHeightResolution, 
	double dPupilDistance)
{
	/*

	[x]			[Row_in]		[Row_out]
	[y] ~= TIn*	[Col_in] ~=Tout*[Col_out]
	[1]			[1     ]		[	   1]

	[Row_in]					[Row_out]	[A00 A01 A02]	[Row_out]
	[Col_in] ~= Inv(TIn)*TOut*	[Col_out] =	[A10 A11 A12] *	[Col_out]
	[1	   ]					[1		]	[A20 A21 1	]	[1		]

	[Col_in]	[A11 A10 A12]	[Col_out]
	[Row_in] ~=	[A01 A00 A02] *	[Row_out]
	[1	   ]	[A21 A20 1	]	[1		]
	*/

	// Validation check (only for 8-bit image)
	if(_bytesPerPixel != 3) return(false);
	
	// Create tansform matrix from (Col_out, Row_out) to (Col_in, Row_in)
	ImgTransform tIn_inv = pImgIn->GetTransform().Inverse();
	ImgTransform t = tIn_inv * _thisToWorld;
	double dTemp[3][3];
	t.GetMatrix(dTemp);
	double dT[3][3];

	dT[0][0] = dTemp[1][1];
	dT[0][1] = dTemp[1][0];
	dT[0][2] = dTemp[1][2];
	
	dT[1][0] = dTemp[0][1];
	dT[1][1] = dTemp[0][0];
	dT[1][2] = dTemp[0][2];

	dT[2][0] = dTemp[2][1];
	dT[2][1] = dTemp[2][0];
	dT[2][2] = dTemp[2][2];

	double dPerpendicalPixelX = (pImgIn->Columns()-1)/2.0; 
	double dPerpendicalPixelY = (pImgIn->Rows()-1)/2.0;
	if(dT[2][0]!=0 || dT[2][1] != 0) // projective transform
	{
		pair<double, double> perpendicalPixel;
		perpendicalPixel = pImgIn->CalPerpendicalPoint(dPupilDistance);
		dPerpendicalPixelX = perpendicalPixel.first;
		dPerpendicalPixelY = perpendicalPixel.second;
	}
	
	// Image morph
	ColorImageMorphWithHeight(
		pImgIn->GetBuffer(), pImgIn->PixelRowStride(),
		pImgIn->Columns(), pImgIn->Rows(),
		_buffer, ByteRowStride(),
		roi.FirstColumn, roi.FirstRow,
		roi.Columns(), roi.Rows(),
		dT,
		pHeightImg->GetBuffer(), pHeightImg->PixelRowStride(),
		dHeightResolution, dPupilDistance,
		dPerpendicalPixelX, dPerpendicalPixelY);

	return(true);
}