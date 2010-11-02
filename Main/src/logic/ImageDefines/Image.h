#pragma once
#include "STL.h"
#include "ImgTransform.h"
#include "UIRect.h"

/*
	Describes an image for stitching

	An Image object can either have a pointer
	to an existing buffer or have its own buffer
	The only difference is if the buffer memory
	gets deallocated when the Image object destructs

	transforms map (row, col) in image space to (x,y) in world space
*/
class Image
{
public:
	// Construcor and configuration file
	Image();
	Image(
		int iColumns, 
		int iRows, 
		int iStride,					// In pixels
		unsigned iDepth,				// Bytes per pixel
		ImgTransform nominalTrans,		// used for overlap calculation before stitching
		ImgTransform actualTrans,		// Stitching results
		unsigned char *buffer = NULL);

	Image(const Image& b);		
	void operator=(const Image& b);	
	virtual ~Image();

	void Configure(	
		int iColumns, 
		int iRows, 
		int iStride,		
		unsigned iDepth,
		unsigned char *buffer = NULL);	

	void Configure(	
		int iColumns, 
		int iRows, 
		int iStride,
		unsigned iDepth,
		ImgTransform nominalTrans,
		ImgTransform actualTrans,
		unsigned char *buffer = NULL);

	// Get/set functions
	ImgTransform		GetTransform() {return _thisToWorld;};
	void				SetTransform(const ImgTransform t) {_thisToWorld = t;};
	ImgTransform		GetNominalTransform(){return _nominalTrans;};
	void				SetNorminalTransform(const ImgTransform t) {_nominalTrans = t;};

	unsigned char*		GetBuffer(){return _buffer;};	
	unsigned char*		GetBuffer(unsigned int row, unsigned col);
	void				SetBuffer(unsigned char* buf);

	bool				HasOwnBuffer() {return _IOwnMyOwnBuffer;};

	short				GetBytesPerPixel(){return _bytesPerPixel;};
	unsigned int		Rows(){return _rows;};
	unsigned int		Columns(){return _columns;};
	unsigned int		ByteRowStride(){return _pixelRowStride*GetBytesPerPixel();};
	unsigned int		PixelRowStride(){return _pixelRowStride;};
	unsigned int		BufferSizeInBytes(){return ByteRowStride()*Rows();};
	
	pair<double,double> ImageCenter( ) const;
	double				CenterX() const;
	double				CenterY() const;
	double				PixelSizeX() const;
	double				PixelSizeY() const;
	double				LengthX() const;
	double				LengthY() const;
	
	// write all zeros to the buffer
	void				ZeroBuffer();
	bool				CreateOwnBuffer();
	void				DeleteBufferIfOwner();

	// Maps (dx,dy) in world space, (Row, col) in image space
	pair<double,double> ImageToWorld(double row, double col ) const;
	void				ImageToWorld(double row, double col, double* pdx, double* pdy) const;

	pair<double,double> WorldToImage(double dx, double dy ) const;
	void				WorldToImage(double dx, double dy, double* pdRow, double* pdCol) const;

	DRect				GetBoundBoxInWorld();

protected:

	// Image size, stride and depth
	unsigned int		_rows;
	unsigned int		_columns;
	unsigned int		_pixelRowStride;
	short				_bytesPerPixel;

	bool				_IOwnMyOwnBuffer;	// Whether image class create and maintain image buffer 
	unsigned char*		_buffer;			// Image buffer
	
	ImgTransform		_thisToWorld;		// Calculated transform after stitching is done
	ImgTransform		_nominalTrans;		// Nominal transform used for overlap calculation before stitching is done
};