#pragma once

#include "Image.h"

/*
	A MosaicImage is a collection of points of images for the same type for a fully scan of a panel.
	The panel can be fully covered by Fovs in a mosiac image and Fovs of image can be arranged by rows and columns

	The MosaicImage is a runtime object, meaning it
	also contains information about which images have been acquired
	or not.
*/

class MosaicImage
{
public:
	MosaicImage(unsigned int iIndex, unsigned int iSizeX, unsigned int iSizeY, bool bUseCad);
	~MosaicImage(void);

	void AddImagePtr(Image* pImage, unsigned int iPosX, unsigned int iPosY);

	Image* GetImagePtr(unsigned int iPosX, unsigned int iPosY) const;
	bool IsImageAcquired(unsigned int iPosX, unsigned int iPosY) const;
	bool IsAcquisitionCompleted() const;

	Image* GetMaskImagePtr(unsigned int iPosX, unsigned int iPosY) const;

	unsigned int NumImages() const {return(_iSizeX * _iSizeY);};
	unsigned int NumImInX() const {return(_iSizeX);};
	unsigned int NumImInY() const {return(_iSizeY);};
	bool UseCad() const {return(_bUseCad);};

	void ImageLineCentersX(double* pdCenX) const;
	void ImageLineCentersY(double* pdCenY) const;

	bool PrepareMaskImages();

	void Reset();

private:	
	unsigned int _iIndex;			// Mosaic image's index
	unsigned int _iSizeX;			// The number of images in column
	unsigned int _iSizeY;			// The number of images in row
	bool _bUseCad;					// Whether need correlation with CAD
	
	Image** _ImagePtrs;				// An array of image points
	bool*	_bImagesAcquired;		// An array of whether image is acquired/added

	unsigned int _iNumImageAcquired;// Number of acquired/added images

	// For mask images
	bool _bIsMaskImgValid;			// Flag of whether mask images are valid to use
	Image* _maskImages;
};

