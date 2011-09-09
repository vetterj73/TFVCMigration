#include "Image.h"
#include "Utilities.h"
#include "Bitmap.h"

#pragma region constructor and configuration
Image::Image(unsigned int iBytePerPixel) 
{
	_rows			= 0;
	_columns		= 0;
	_pixelRowStride = 0;
	_bytesPerPixel	= iBytePerPixel;
	_IOwnMyOwnBuffer= false;	
	_buffer			=0;
}

Image::Image(
		int iColumns, 
		int iRows, 
		int iStride,
		unsigned int iBytePerPixel,
		ImgTransform nominalTrans,
		ImgTransform actualTrans,
		bool bCreateOwnBuffer,
		unsigned char *buffer)
{
	// To avoid crash ins some scenarios
	_IOwnMyOwnBuffer = false;
	_buffer = 0;

	_bytesPerPixel=iBytePerPixel;
	
	Configure(	
		iColumns, 
		iRows, 
		iStride,
		nominalTrans,
		actualTrans,
		bCreateOwnBuffer,
		buffer);
}

Image::Image(const Image& b)
{
	*this = b;
}
		
void Image::operator=(const Image& b)
{
	_rows				= b._rows;
	_columns			= b._columns;
	_pixelRowStride		= b._pixelRowStride;	
	_bytesPerPixel		= b._bytesPerPixel;
	
	_thisToWorld		= b._thisToWorld;
	_nominalTrans		= b._nominalTrans;

	DeleteBufferIfOwner();

	_IOwnMyOwnBuffer	= b._IOwnMyOwnBuffer;
	if( _IOwnMyOwnBuffer )
	{
		_buffer			= new unsigned char[BufferSizeInBytes()];	
		::memcpy(_buffer, b._buffer, BufferSizeInBytes());
	}
	else
	{
		_buffer			= b._buffer;
	}
}

Image::~Image()
{
	DeleteBufferIfOwner();
}

void Image::Configure(
	int iColumns, 
	int iRows, 
	int iStride,
	bool bCreateOwnBuffer,
	unsigned char *buffer)
{
	_rows				= iRows;
	_columns			= iColumns;
	_pixelRowStride		= iStride;

	DeleteBufferIfOwner();
	
	if(bCreateOwnBuffer)
	{
		_buffer	= new unsigned char[BufferSizeInBytes()];	
		_IOwnMyOwnBuffer = true;
	} 
	else
		SetBuffer(buffer);
}

void Image::Configure(	
		int iColumns, 
		int iRows, 
		int iStride,
		ImgTransform nominalTrans,
		ImgTransform actualTrans,
		bool bCreateOwnBuffer,
		unsigned char *buffer)
{
	Configure(
		iColumns, 
		iRows, 
		iStride,
		bCreateOwnBuffer,
		buffer);

	_nominalTrans = nominalTrans;
	_thisToWorld = actualTrans;
}

#pragma endregion

#pragma region buffer operations

// Get buffer point at certain location
unsigned char*	Image::GetBuffer(unsigned col, unsigned int row) const
{
	return(GetBuffer() + PixelRowStride()*row + GetBytesPerPixel()*col);
}

void Image::SetBuffer(unsigned char* buf)
{
	DeleteBufferIfOwner();
	_buffer = buf;
}

bool Image::CreateOwnBuffer()
{
	if(_IOwnMyOwnBuffer) return true;

	_buffer	= new unsigned char[BufferSizeInBytes()];	
	if(_buffer == NULL)
		return false;
	else
	{
		_IOwnMyOwnBuffer = true;
		return true;
	}
}

void Image::DeleteBufferIfOwner()
{
	if(_IOwnMyOwnBuffer && _buffer!=NULL)
	{
		delete[] _buffer;
		_buffer = NULL;
	}

	_IOwnMyOwnBuffer = false;
}

// clear the image
void Image::ZeroBuffer()
{
	if(_buffer != NULL)
	::memset(_buffer, 0, BufferSizeInBytes());
}

// Save image to disc
bool Image::Save(string sFileName)
{
	Bitmap* pBmp = Bitmap::NewBitmapFromBuffer(
		_rows, _columns, ByteRowStride(), _buffer, _bytesPerPixel*8);
	
	if(pBmp == NULL) return(false);
		
	pBmp->write(sFileName);

	delete pBmp;

	return(true);
}

#pragma endregion

#pragma region Transform related
// transforms map (row, col) in image space to (x, y) in world space

// Map (row, col) in image space to (x, y) in world space
pair<double, double> Image::ImageToWorld(double row, double col) const
{
	pair<double, double> pare;
	_thisToWorld.Map(row, col, &pare.first, &pare.second);

	return pare;
}

// Map (row, col) in image space to (x, y) in world space
void Image::ImageToWorld(double row, double col, double* pdx, double* pdy) const
{
	_thisToWorld.Map(row, col, pdx, pdy);
}

// Map (x, y) in world space to (row, col) in image space
pair<double, double> Image::WorldToImage(double dx, double dy)
{
	pair<double, double> pare;
	_thisToWorld.InverseMap(dx, dy, &pare.first, &pare.second);

	return pare;
}

// Map (x, y) in world space to (row, col) in image space
void Image::WorldToImage(double dx, double dy, double* pdRow, double* pdCol)	
{
	_thisToWorld.InverseMap(dx, dy, pdRow, pdCol); 
}

// Image Center in world space
pair<double, double> Image::ImageCenter( ) const
{
	return ImageToWorld( 0.5*(_rows-1.0), 0.5*(_columns-1.0));
}

// Image Center in x of world space
double Image::CenterX() const
{
	return ImageCenter().first;
}

// Image Center in y of world space
double Image::CenterY() const
{
	return ImageCenter().second;
}

// FOV pixel size in x of world space
double Image::PixelSizeX() const
{
	return(_thisToWorld.GetItem(0,0));
}

// FOV pixel size in y of world space
double Image::PixelSizeY() const
{
	return(_thisToWorld.GetItem(1,1));
}

// FOV length in x of world space
double Image::LengthX() const
{
	return _rows*PixelSizeX();
}

// FOV length in y of world space
 double Image::LengthY() const
{
	return _columns*PixelSizeY();
}

DRect Image::GetBoundBoxInWorld() const
{
	DRect rect;

	pair<double, double> topLeft = ImageToWorld(0, 0);
	pair<double, double> topRight = ImageToWorld(0, _columns-1);
	pair<double, double> bottomLeft = ImageToWorld(_rows-1, 0);
	pair<double, double> bottomRight = ImageToWorld(_rows-1, _columns-1);
 
	rect.xMin = topLeft.first>topRight.first ? topLeft.first:topRight.first;
	rect.xMax = bottomLeft.first<bottomRight.first ? bottomLeft.first:bottomRight.first;
	rect.yMin = topLeft.second>bottomLeft.second ? topLeft.second:bottomLeft.second;
	rect.yMax = topRight.second<bottomRight.second ? topRight.second:bottomRight.second;

	return rect;
}

// This image's ROI content is mapped from pImgIn
bool Image::MorphFrom(const Image* pImgIn, UIRect roi)
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
	if(_bytesPerPixel != 1) return(false);
	
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

	// Image morph
	ImageMorph(
		pImgIn->GetBuffer(), pImgIn->PixelRowStride(),
		pImgIn->Columns(), pImgIn->Rows(),
		_buffer, _pixelRowStride,
		roi.FirstColumn, roi.FirstRow,
		roi.Columns(), roi.Rows(),
		dT);

	return(true);
}

// This image's ROI content is mapped from pImgIn
bool Image::MorphFromWithHeight(
	Image* pImgIn, 
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
	if(_bytesPerPixel != 1) return(false);
	
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
	ImageMorphWithHeight(
		pImgIn->GetBuffer(), pImgIn->PixelRowStride(),
		pImgIn->Columns(), pImgIn->Rows(),
		_buffer, _pixelRowStride,
		roi.FirstColumn, roi.FirstRow,
		roi.Columns(), roi.Rows(),
		dT,
		pHeightImg->GetBuffer(), pHeightImg->PixelRowStride(),
		dHeightResolution, dPupilDistance,
		dPerpendicalPixelX, dPerpendicalPixelY);

	return(true);
}

// Calcualte the pixel corresponding to the point in the panel surface 
// that its connection with camera center is vertical to panel surface
// This is for projective transform only, 
// Assume without perspective angle, this point is the image center
// return in (cloumn Row)
pair<double,double> Image::CalPerpendicalPoint(double dPupilDistance)
{
	// The transform
	double a[3][3];
	_thisToWorld.GetMatrix(a); 
	
	// Image center
	double xo, yo;
	ImageToWorld((Rows()-1)/2.0, (Columns()-1)/2.0, &xo, &yo);

	// transform from world to image
	double dT[3][3];
	_thisToWorld.GetInvertMatrix(dT);	
	double dMinValue = 1e-10;
	
	// X dimension calculation
	double temp = dT[2][0];
	if(temp<dMinValue) temp = dMinValue; 
	
	double xm = -(dT[2][1]*yo+1)/temp;
	double dSinThetaX = dPupilDistance/(xm-xo);

	// Y dimension calculation
	temp = dT[2][1];
	if(temp<dMinValue) temp = dMinValue;

	double ym = -(dT[2][0]*xo+1)/temp;
	double dSinThetaY = dPupilDistance/(ym-yo);

	// calculate vertic point
	double dPx = xo + dSinThetaX*dPupilDistance;
	double dPy = yo + dSinThetaY*dPupilDistance;

	pair<double, double> perpendicalPixel;
	WorldToImage(dPx, dPy, &perpendicalPixel.second, &perpendicalPixel.first); 

	return(perpendicalPixel);
}

#pragma endregion

