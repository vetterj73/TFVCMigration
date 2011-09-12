#pragma once
#include "image.h"
#include "Utilities.h"

// Only support BGR and YCrCb color type only 

class ColorImage : public Image
{
public:
	ColorImage(COLORSTYLE colorStyle, bool bChannelStoredSeperate);
	~ColorImage(void);

	bool DemosaicFrom(const Image* bayerImg, BayerType type);
	bool DemosiacFrom(unsigned char* pBayerBuf, int iColums, int iRows, int iSpan, BayerType type);
	bool ColorMorphFrom(const ColorImage* pImgIn, UIRect roi);
	bool  ColorImage::ColorMorphFromWithHeight(
		ColorImage* pImgIn, 
		UIRect roi,
		const Image* pHeightImg, 
		double dHeightResolution, 
		double dPupilDistance);

	COLORSTYLE GetColorStyle() {return _colorStyle;};
	void SetColorStyle(COLORSTYLE value);

	bool IsChannelStoredSeperated() {return _bChannelStoredSeperate;};
	void SetChannelStoreSeperated(bool bValue);

	bool IsChannelStoredSeperate() {return _bChannelStoredSeperate;};

private:
	COLORSTYLE _colorStyle;
	bool _bChannelStoredSeperate;
};
