#pragma once
#include "image.h"
#include "Utilities.h"

// Only support RGB and YCrCb color type only 

class ColorImage : public Image
{
public:
	ColorImage(COLORSTYLE colorStyle = RGB, bool bChannelStoredSeperate = false);
	~ColorImage(void);

	bool DemosaicFrom(const Image* bayerImg, BayerType type);
	bool DemosiacFrom(unsigned char* pBayerBuf, int iColums, int iRows, int iSpan, BayerType type);
	bool ColorMorphFrom(const ColorImage* pImgIn, UIRect roi);

	COLORSTYLE GetColorStyle() {return _colorStyle;};
	void SetColorStyle(COLORSTYLE value);

	bool IsChannelStoredSeperated() {return _bChannelStoredSeperate;};
	void SetChannelStoreSeperated(bool bValue);

	bool IsChannelStoredSeperate() {return _bChannelStoredSeperate;};

private:
	COLORSTYLE _colorStyle;
	bool _bChannelStoredSeperate;
};

