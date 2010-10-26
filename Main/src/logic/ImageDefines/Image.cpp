#include "Image.h"

#pragma region constructor and configuration
Image::Image() 
{
	_rows=0;
	_columns=0;
	_pixelRowStride=0;
	_bytesPerPixel=0;
	_IOwnMyOwnBuffer= false;	
	_buffer=0;
}

Image::Image(
		int iColumns, 
		int iRows, 
		int iStride,
		unsigned iDepth,
		ImgTransform nominalTrans,
		ImgTransform actualTrans,
		unsigned char *buffer)
{
	Configure(	
		iColumns, 
		iRows, 
		iStride,
		iDepth,
		nominalTrans,
		actualTrans,
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

}

void Image::Configure(
	int iColumns, 
	int iRows, 
	int iStride,
	unsigned iDepth,
	unsigned char *buffer)
{
	_rows				= iRows;
	_columns			= iColumns;
	_pixelRowStride		= iStride;
	_bytesPerPixel		= iDepth;

	DeleteBufferIfOwner();
	
	if(buffer == NULL)
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
		unsigned iDepth,
		ImgTransform nominalTrans,
		ImgTransform actualTrans,
		unsigned char *buffer)
{
	Configure(
		iColumns, 
		iRows, 
		iStride,
		iDepth,
		buffer);

	_nominalTrans = nominalTrans;
	_thisToWorld = actualTrans;
}

#pragma endregion

#pragma region buffer operations

void Image::SetBuffer(unsigned char* buf)
{
	DeleteBufferIfOwner();
	_buffer = buf;
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
	::memset(_buffer, 0, BufferSizeInBytes());
}

#pragma endregion

#pragma region Transform related
// transforms map (row, col) in image space to (x, y) in world space

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
pair<double, double> Image::WorldToImage(double dx, double dy ) const
{
	pair<double, double> pare;
	_thisToWorld.InverseMap(dx, dy, &pare.first, &pare.second);

	return pare;
}

// Map (x, y) in world space to (row, col) in image space
void Image::WorldToImage(double dx, double dy, double* pdRow, double* pdCol) const	
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

#pragma endregion