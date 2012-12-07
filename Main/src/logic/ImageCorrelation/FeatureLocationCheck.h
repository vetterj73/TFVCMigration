#pragma once
#include "Panel.h"

// This class is to find the locations of fiducials on the panel (image)
class Image;
class FeatureLocationCheck
{
public:
	FeatureLocationCheck(Panel* pPanel);
	~FeatureLocationCheck(void);

	bool CheckFeatureLocation(Image* pImage, double d[]);

private:
	Panel* _pPanel;
	int* _piTemplateIds;
	double _dSearchExpansion;

	// for debug
	int _iCycleCount;
	Image* _pFidImages;

};

class ImageFidAligner
{
public: 
	ImageFidAligner(Panel* pPanel);
	~ImageFidAligner();

	bool CalculateTransform(Image* pImage, double t[3][3], double* pZ = NULL);

	bool MorphImage(Image* pImgIn, Image* pImgOut, double* pZ = NULL);

private:
	Panel* _pPanel;
	FeatureLocationCheck* _pFidFinder;
	Image* _pMorphedImage;
};

