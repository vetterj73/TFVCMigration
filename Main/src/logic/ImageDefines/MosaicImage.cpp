#include "MosaicImage.h"

MosaicImage::MosaicImage(unsigned int iIndex, unsigned int iSizeX, unsigned int iSizeY, bool bUseCad)
{
	_iIndex = iIndex;
	_iSizeX = iSizeX;
	_iSizeY = iSizeY;
	_bUseCad = bUseCad;

	_ImagePtrs = new Image*[NumImages()];
	_bImagesAcquired = new bool[NumImages()];

	_maskImages = new Image[NumImages()];

	Reset();
}

void MosaicImage::Reset()
{
	for(unsigned int i=0; i<NumImages(); i++)
		_bImagesAcquired[i] = false;

	_bIsMaskImgValid = false;

	_iNumImageAcquired = 0;
}

MosaicImage::~MosaicImage(void)
{
	delete [] _ImagePtrs;
	delete [] _bImagesAcquired;
	delete [] _maskImages;
}

// Add an image point of certain position to mosaic image
void MosaicImage::AddImagePtr(	
	Image* pImage, 
	unsigned int iPosX, 
	unsigned int iPosY)
{
	unsigned int iPos = iPosY*_iSizeX+ iPosX;
	_ImagePtrs[iPos] = pImage;

	if(!_bImagesAcquired[iPos])
	{
		_bImagesAcquired[iPos] = true;
		_iNumImageAcquired++;
	}
}

//Get image point in certain position
Image* MosaicImage::GetImagePtr(unsigned int iPosX, unsigned int iPosY) const
{
	unsigned int iPos = iPosY*_iSizeX+ iPosX;
	return(_ImagePtrs[iPos]);
}

// Return true if a image in certain position is acquired/added
bool MosaicImage::IsImageAcquired(unsigned int iPosX, unsigned int iPosY) const
{
	unsigned int iPos = iPosY*_iSizeX+ iPosX;
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

// Get average center in X of image Columns
void MosaicImage::ImageLineCentersX(double* pdCenX) const
{
	for(unsigned int ix=0; ix<_iSizeX; ix++)
	{
		pdCenX[ix] = 0;
		for(unsigned int iy=0; iy<_iSizeY; iy++)
		{
			pdCenX[ix] += GetImagePtr(ix, iy)->CenterX();
		}
		pdCenX[ix] /= _iSizeY;
	}
}

// Get average center in Y of image rows
void MosaicImage::ImageLineCentersY(double* pdCenY) const
{
	for(unsigned int iy=0; iy<_iSizeY; iy++)
	{
		pdCenY[iy] = 0;
		for(unsigned int ix=0; ix<_iSizeX; ix++)
		{
			pdCenY[iy] += GetImagePtr(ix, iy)->CenterY();
		}
		pdCenY[iy] /= _iSizeX;
	}
}

// Prepare Mask images to use (validate mask images)
bool MosaicImage::PrepareMaskImages()
{
	// Validation check
	if(!IsAcquisitionCompleted()) return(false);

	for(unsigned int i=0 ; i<NumImages(); i++)
	{
		_maskImages[i] = *(_ImagePtrs[i]);
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

	unsigned int iPos = iPosY*_iSizeX+ iPosX;
	return(_ImagePtrs[iPos]);
}

		







