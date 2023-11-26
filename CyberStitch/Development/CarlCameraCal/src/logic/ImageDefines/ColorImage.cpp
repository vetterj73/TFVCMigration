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
				iTemp[2] = (int)pLine[iAddress[0]] + ((int)pLine[iAddress[1]]-128);									// R
				iTemp[1] = (int)pLine[iAddress[0]] - ((int)pLine[iAddress[1]]-128 + (int)pLine[iAddress[2]]-128)/2;	// G
				iTemp[0] = (int)pLine[iAddress[0]] + ((int)pLine[iAddress[2]]-128);	// B
			}
			// BGR to YCrCb conversion
			/* 
				Cr = 3/4*R - 1/2*G - 1/4*B
				Cb = -1/4*R -1/2*g + 3/4*B
				When R, G, B in range of [0, 255]
				Cr and Cb in range of [-255*3/4, 255*3/4] = [-191, 191], Which is out or range of 2^8 [-128, 127]
				If scale Cr and Cb down by 2, RGB image converted from YCrCb will lose some resolution in intensity.
				For circuit board, the chance of Cr and Cb out of [-128, 127] is rare
				Therefore, in order to save storage, Cr and Cb is clip into [-128, 127]+128 = [0, 255]
			*/
			if(_colorStyle ==  BGR && value == YCrCb)
			{
				iTemp[0] = (pLine[iAddress[2]] + pLine[iAddress[1]]*2 + pLine[iAddress[0]])>>2;	// Y	
				iTemp[1] = pLine[iAddress[2]] - iTemp[0] + 128;										// Cr
				iTemp[2] = pLine[iAddress[0]] - iTemp[0] + 128;									// Cb
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

// Convert Color image into greyscal image, 
// greyscale image will be configurated and its buffer will be allocated 
bool ColorImage::Color2Luminance(Image* pGreyImg)
{
	// Configure and allocate buffer for greyscale image
	pGreyImg->Configure(_columns, _rows, _columns, _nominalTrans, _thisToWorld, true);

	unsigned char* pColorLine = _buffer;
	unsigned char* pGreyLine = pGreyImg->GetBuffer();
	for(unsigned int iy=0; iy<_rows; iy++) 
	{
		for(unsigned int ix=0; ix<_columns; ix++)
		{
			unsigned char Y = (pColorLine[ix*_bytesPerPixel] + pColorLine[ix*_bytesPerPixel+1]*2 + pColorLine[ix*_bytesPerPixel+2])>>2;	// Y	
			pGreyLine[ix] = Y;
		}
		pColorLine += ByteRowStride();		
		pGreyLine += pGreyImg->ByteRowStride();
	}

	return(true);
}


