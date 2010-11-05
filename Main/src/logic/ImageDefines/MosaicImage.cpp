#include "MosaicImage.h"

MosaicImage::MosaicImage()
{
	_images = NULL;
	_bImagesAcquired = NULL;
	_maskImages = NULL;
}

MosaicImage::MosaicImage(
	unsigned int iIndex, 
	unsigned int iNumImgX, 
	unsigned int iNumImgY,
	unsigned int iImColumns,
	unsigned int iImRows,
	unsigned int iImStride,
	bool bUseCad)
{
	Config(
		iIndex, 
		iNumImgX, 
		iNumImgY,
		iImColumns,
		iImRows,
		iImStride,
		bUseCad);
}

MosaicImage::~MosaicImage(void)
{
	if(_images != NULL) delete [] _images;
	if(_bImagesAcquired != NULL) delete [] _bImagesAcquired;
	if(_maskImages != NULL) delete [] _maskImages;
}

void MosaicImage::Config(
	unsigned int iIndex, 
	unsigned int iNumImgX, 
	unsigned int iNumImgY,
	unsigned int iImColumns,
	unsigned int iImRows,
	unsigned int iImStride,
	bool bUseCad)
{
	_iIndex = iIndex;
	_iNumImgX = iNumImgX;
	_iNumImgY = iNumImgY;
	_bUseCad = bUseCad;

	_images = new Image[NumImages()];
	_maskImages = new Image[NumImages()];
	unsigned int iBytePerPixel = 1;
	bool bCreatOwnBuffer = false;
	for(unsigned int i=0; i<NumImages(); i++)
	{
		_images[i].Configure(iImColumns, iImRows, iImStride, iBytePerPixel, bCreatOwnBuffer);
		_maskImages[i].Configure(iImColumns, iImRows, iImStride, iBytePerPixel, bCreatOwnBuffer);
	}
	
	_bImagesAcquired = new bool[NumImages()];
	ResetForNextPanel();
}




// Reset to prepare the next panel alignment
void MosaicImage::ResetForNextPanel()
{
	for(unsigned int i=0; i<NumImages(); i++)
	{
		_images[i].SetTransform(_images[i].GetNominalTransform());
		_maskImages[i].SetTransform(_maskImages[i].GetNominalTransform());
	}	
	
	for(unsigned int i=0; i<NumImages(); i++)
		_bImagesAcquired[i] = false;	
	
	_iNumImageAcquired = 0;

	_bIsMaskImgValid = false;
}


// Set both nominal and regular transform for an image in certain position
void MosaicImage::SetImageTransforms(ImgTransform trans, unsigned int iPosX, unsigned int iPosY)
{
	unsigned int iPos = iPosY*_iNumImgX+ iPosX;

	if(!_bImagesAcquired[iPos])
	{
		_images[iPos].SetNorminalTransform(trans);
		_images[iPos].SetTransform(trans);
	}
}

// Add the buffer for an image in certain position
void MosaicImage::AddImageBuffer(unsigned char* pBuffer, unsigned int iPosX, unsigned int iPosY)
{
	unsigned int iPos = iPosY*_iNumImgX+ iPosX;

	if(!_bImagesAcquired[iPos])
	{
		_images[iPos].SetBuffer(pBuffer);
		_bImagesAcquired[iPos] = true;
		_iNumImageAcquired++;
	}
}

//Get image point in certain position
Image* MosaicImage::GetImagePtr(unsigned int iPosX, unsigned int iPosY) const
{
	unsigned int iPos = iPosY*_iNumImgX+ iPosX;
	return(&_images[iPos]);
}

// Return true if a image in certain position is acquired/added
bool MosaicImage::IsImageAcquired(unsigned int iPosX, unsigned int iPosY) const
{
	unsigned int iPos = iPosY*_iNumImgX+ iPosX;
	return(_bImagesAcquired[iPos]);
}

// Return true if all images for this mosaic are collected/acquired/added
bool MosaicImage::IsAcquisitionCompleted() const
{
	if(_iNumImageAcquired == NumImages())
		return(true);
	else
		return(false);
}

// Conside images are arrranged in x-y grids
// Centers in X direction of grid 
void MosaicImage::ImageGridXCenters(double* pdCenX) const
{
	for(unsigned int ix=0; ix<_iNumImgX; ix++)
	{
		pdCenX[ix] = 0;
		for(unsigned int iy=0; iy<_iNumImgY; iy++)
		{
			pdCenX[ix] += GetImagePtr(ix, iy)->CenterX();
		}
		pdCenX[ix] /= _iNumImgY;
	}
}

// Conside images are arrranged in x-y grids
// Centers in Y direction of grid 
void MosaicImage::ImageGridYCenters(double* pdCenY) const
{
	for(unsigned int iy=0; iy<_iNumImgY; iy++)
	{
		pdCenY[iy] = 0;
		for(unsigned int ix=0; ix<_iNumImgX; ix++)
		{
			pdCenY[iy] += GetImagePtr(ix, iy)->CenterY();
		}
		pdCenY[iy] /= _iNumImgX;
	}
}

// Prepare Mask images to use (validate mask images)
bool MosaicImage::PrepareMaskImages()
{
	// Validation check
	if(!IsAcquisitionCompleted()) return(false);

	for(unsigned int i=0 ; i<NumImages(); i++)
	{
		_maskImages[i].SetTransform(_images[i].GetTransform());
		_maskImages[i].CreateOwnBuffer();
	}

	_bIsMaskImgValid = true;

	return true;
}

// Get a mask image point in certain position
// return NULL if it is not valid
Image* MosaicImage::GetMaskImagePtr(unsigned int iPosX, unsigned int iPosY) const
{
	// Validation check
	if(!_bIsMaskImgValid)
		return NULL;

	unsigned int iPos = iPosY*_iNumImgX+ iPosX;
	return(&_maskImages[iPos]);
}

		







