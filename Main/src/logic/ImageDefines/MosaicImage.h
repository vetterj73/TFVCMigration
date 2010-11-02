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
	MosaicImage(unsigned int iSizeX, unsigned int iSizeY);
	~MosaicImage(void);

	void AddImagePtr(Image* pImage, unsigned int iPosX, unsigned int iPosY);

	Image* GetImagePtr(unsigned int iPosX, unsigned int iPosY);
	bool IsImageAcquired(unsigned int iPosX, unsigned int iPosY);
	bool IsAcquisitionCompleted();

	unsigned int NumImages() {return(_iSizeX * _iSizeY);};
	unsigned int NumImInX() {return(_iSizeX);};
	unsigned int NumImInY() {return(_iSizeY);};

	void ImageLineCentersX(double* pdCenX);
	void ImageLineCentersY(double* pdCenY);

	void Reset();

private:
	Image** _ImagePtrs;				// An array of image points
	bool*	_bImagesAcquired;		// An array of whether image is acquired/added

	unsigned int _iNumImageAcquired;// Number of acquired/added images

	unsigned int _iSizeX;			// The number of images in column
	unsigned int _iSizeY;			// The number of images in row

	// For mask images
	bool _bHasMaskImages;
	Image* _maskImages;
};

