#include "MosaicImage.h"

MosaicImage::MosaicImage(unsigned int iSizeX, unsigned int iSizeY)
{
	_iSizeX = iSizeX;
	_iSizeY = iSizeY;

	_ImagePtrs = new Image*[NumImages()];
	_bImagesAcquired = new bool[NumImages()];

	Reset();
}

void MosaicImage::Reset()
{
	for(unsigned int i=0; i<NumImages(); i++)
		_bImagesAcquired[i] = false;

	_iNumImageAcquired = 0;
}

MosaicImage::~MosaicImage(void)
{
	delete [] _ImagePtrs;
	delete [] _bImagesAcquired;
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
Image* MosaicImage::GetImagePtr(unsigned int iPosX, unsigned int iPosY)
{
	unsigned int iPos = iPosY*_iSizeX+ iPosX;
	return(_ImagePtrs[iPos]);
}

// Return true if a image in certain position is acquired/added
bool MosaicImage::IsImageAcquired(unsigned int iPosX, unsigned int iPosY)
{
	unsigned int iPos = iPosY*_iSizeX+ iPosX;
	return(_bImagesAcquired[iPos]);
}

// Return true if all images for this mosaic are collected/acquired/added
bool MosaicImage::IsAcquisitionCompleted()
{
	if(_iNumImageAcquired == NumImages())
		return(true);
	else
		return(false);
}







