#pragma once

#include "Image.h"

/*
	A MosaicImage is a collection of points of images for the same type for a fully scan of a panel.
	The panel can be fully covered by Fovs in a mosiac image and Fovs of image can be arranged by rows and columns

	The MosaicImage is a runtime object, meaning it	also contains information about which images have been acquired or not.
*/

class MosaicImage
{
public:
	MosaicImage(
		unsigned int iIndex,		// Index of mosaic image
		unsigned int iNumImgX,		// Number of images in x direction
		unsigned int iNumImgY,		// Number of images in y direction
		unsigned int iImColumns,	// Columns of each image
		unsigned int iImRows,		// Rows of each image
		unsigned int iImStride,		// Stride of each image
		bool bUseCad);				// Flag of whether Cad is used for alignment

	~MosaicImage(void);	
	
	void ResetForNextPanel();

	void SetImageTransforms(ImgTransform trans, unsigned int iPosX, unsigned int iPosY);
	void AddImageBuffer(unsigned char* pBuffer, unsigned int iPosX, unsigned int iPosY);

	Image* GetImagePtr(unsigned int iPosX, unsigned int iPosY) const;
	bool IsImageAcquired(unsigned int iPosX, unsigned int iPosY) const;
	bool IsAcquisitionCompleted() const;

	Image* GetMaskImagePtr(unsigned int iPosX, unsigned int iPosY) const;
	bool PrepareMaskImages();

	unsigned int NumImages() const {return(_iNumImgX * _iNumImgY);};
	unsigned int NumImInX() const {return(_iNumImgX);};
	unsigned int NumImInY() const {return(_iNumImgY);};
	
	bool UseCad() const {return(_bUseCad);};

	void ImageGridXCenters(double* pdCenX) const;
	void ImageGridYCenters(double* pdCenY) const;
	
private:	
	unsigned int _iIndex;			// Mosaic image's index
	unsigned int _iNumImgX;			// The number of images in column
	unsigned int _iNumImgY;			// The number of images in row
	bool _bUseCad;					// Whether need correlation with CAD
	
	Image*	_images;				// An array of images
	bool*	_bImagesAcquired;		// An array of whether image buffer is acquired/added

	unsigned int _iNumImageAcquired;// Number of acquired/added images

	// For mask images
	bool _bIsMaskImgValid;			// Flag of whether mask images are valid to use
	Image* _maskImages;				// An array fo mask images 
};

