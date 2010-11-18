#include "MosaicImage.h"
#include "Logger.h"

MosaicImage::MosaicImage()
{
	_images = NULL;
	_bImagesAcquired = NULL;
	_maskImages = NULL;
}

MosaicImage::MosaicImage(
	unsigned int iIndex, 
	unsigned int iNumCameras, 
	unsigned int iNumTriggers,
	unsigned int iImColumns,
	unsigned int iImRows,
	unsigned int iImStride,
	bool bUseCad)
{
	Config(
		iIndex, 
		iNumCameras, 
		iNumTriggers,
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
	unsigned int iNumCameras, 
	unsigned int iNumTriggers,
	unsigned int iImColumns,
	unsigned int iImRows,
	unsigned int iImStride,
	bool bUseCad)
{
	_iIndex = iIndex;
	_iNumCameras = iNumCameras;
	_iNumTriggers = iNumTriggers;
	_bUseCad = bUseCad;

	// Create images and mask images
	_images = new Image[NumImages()];
	_maskImages = new Image[NumImages()];
	bool bCreatOwnBuffer = false;
	for(unsigned int i=0; i<NumImages(); i++)
	{
		_images[i].Configure(iImColumns, iImRows, iImStride, bCreatOwnBuffer);
		_maskImages[i].Configure(iImColumns, iImRows, iImStride, bCreatOwnBuffer);
	}
	
	// Create array for image acquisition flags
	_bImagesAcquired = new bool[NumImages()];

	// Rest all flags
	ResetForNextPanel();
}

// Reset to prepare the next panel alignment
void MosaicImage::ResetForNextPanel()
{
	// Reset image transform to nominal one
	for(unsigned int i=0; i<NumImages(); i++)
	{
		_images[i].SetTransform(_images[i].GetNominalTransform());
		_maskImages[i].SetTransform(_maskImages[i].GetNominalTransform());
	}	
	
	// Clean acquiresation flags, count and mask image validation flag
	for(unsigned int i=0; i<NumImages(); i++)
		_bImagesAcquired[i] = false;	
	
	_iNumImageAcquired = 0;

	_bIsMaskImgValid = false;
}


// Set both nominal and regular transform for an image in certain position
void MosaicImage::SetImageTransforms(ImgTransform trans, unsigned int iCamIndex, unsigned int iTrigIndex)
{
	unsigned int iPos = iTrigIndex*_iNumCameras+ iCamIndex;

	if(!_bImagesAcquired[iPos])
	{
		_images[iPos].SetNorminalTransform(trans);
		_images[iPos].SetTransform(trans);
	}
}

// Add the buffer for an image in certain position
void MosaicImage::AddImageBuffer(unsigned char* pBuffer, unsigned int iCamIndex, unsigned int iTrigIndex)
{
	unsigned int iPos = iTrigIndex*_iNumCameras+ iCamIndex;

	if(!_bImagesAcquired[iPos])
	{
		_images[iPos].SetBuffer(pBuffer);
		_bImagesAcquired[iPos] = true;
		_iNumImageAcquired++;
	}
}

//Get image point in certain position
Image* MosaicImage::GetImage(unsigned int iCamIndex, unsigned int iTrigIndex) const
{
	unsigned int iPos = iTrigIndex*_iNumCameras+ iCamIndex;
	return(&_images[iPos]);
}

// Return true if a image in certain position is acquired/added
bool MosaicImage::IsImageAcquired(unsigned int iCamIndex, unsigned int iTrigIndex) const
{
	unsigned int iPos = iTrigIndex*_iNumCameras+ iCamIndex;
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

// Camera centers in Y of world space
void MosaicImage::CameraCentersInY(double* pdCenY) const
{
	for(unsigned int iCam=0; iCam<_iNumCameras; iCam++)
	{
		pdCenY[iCam] = 0;
		for(unsigned int iTrig=0; iTrig<_iNumTriggers; iTrig++)
		{
			pdCenY[iCam] += GetImage(iCam, iTrig)->CenterY();
		}
		pdCenY[iCam] /= _iNumTriggers;
	}
}

// Trigger centesr in  X of world space 
void MosaicImage::TriggerCentersInX(double* pdCenX) const
{
	for(unsigned int iTrig=0; iTrig<_iNumTriggers; iTrig++)
	{
		pdCenX[iTrig] = 0;
		for(unsigned int iCam=0; iCam<_iNumCameras; iCam++)
		{
			pdCenX[iTrig] += GetImage(iCam, iTrig)->CenterX();
		}
		pdCenX[iTrig] /= _iNumCameras;
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
Image* MosaicImage::GetMaskImage(unsigned int iCamIndex, unsigned int iTrigIndex) const
{
	// Validation check
	if(!_bIsMaskImgValid)
		return NULL;

	unsigned int iPos = iTrigIndex*_iNumCameras+ iCamIndex;
	return(&_maskImages[iPos]);
}

		







