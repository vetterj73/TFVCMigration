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
	MosaicImage();

	MosaicImage(
		unsigned int iIndex,		// Index of mosaic image
		unsigned int iNumCameras,	// Number of cameras (images in a row)
		unsigned int iNumTriggers,	// Number of triggers (images in a column)
		unsigned int iImColumns,	// Columns of each image
		unsigned int iImRows,		// Rows of each image
		unsigned int iImStride,		// Stride of each image
		bool bAlignWithCad,
		bool bAlignWithFiducial);		// Flag of whether Cad is used for alignment

	void Config(
		unsigned int iIndex,		
		unsigned int iNumCameras,		
		unsigned int iNumTriggers,		
		unsigned int iImColumns,	
		unsigned int iImRows,		
		unsigned int iImStride,		
		bool bAlignWithCad,
		bool bAlignWithFiducial);				

	~MosaicImage(void);	
	
	void ResetForNextPanel();

	void SetImageTransforms(ImgTransform trans, unsigned int iCamIndex, unsigned int iTrigIndex);
	void AddImageBuffer(unsigned char* pBuffer, unsigned int iCamIndex, unsigned int iTrigIndex);

	Image* GetImage(unsigned int iCamIndex, unsigned int iTrigIndex) const;
	bool IsImageAcquired(unsigned int iCamIndex, unsigned int iTrigIndex) const;
	bool IsAcquisitionCompleted() const;

	Image* GetMaskImage(unsigned int iCamIndex, unsigned int iTrigIndex) const;
	bool PrepareMaskImages();

	unsigned int Index() const {return(_iIndex);};
	unsigned int NumImages() const {return(_iNumCameras * _iNumTriggers);};
	unsigned int NumCameras() const {return(_iNumCameras);};
	unsigned int NumTriggers() const {return(_iNumTriggers);};
	
	bool AlignWithCad() const {return(_bAlignWithCad);};
	bool AlignWithFiducial() const {return(_bAlignWithFiducial);};

	void CameraCentersInY(double* pdCenY) const;
	void TriggerCentersInX(double* pdCenX) const;
	
private:	
	unsigned int _iIndex;			// Mosaic image's index
	unsigned int _iNumCameras;		// Number of cameras (images in a row)
	unsigned int _iNumTriggers;		// Number of triggers (images in a column)
	bool _bAlignWithCad;			// Whether need align with CAD
	bool _bAlignWithFiducial;		// Whether need align with Fiducial
	
	Image*	_images;				// An array of images
	bool*	_bImagesAcquired;		// An array of whether image buffer is acquired/added

	unsigned int _iNumImageAcquired;// Number of acquired/added images

	// For mask images
	bool _bIsMaskImgValid;			// Flag of whether mask images are valid to use
	Image* _maskImages;				// An array fo mask images 
};

