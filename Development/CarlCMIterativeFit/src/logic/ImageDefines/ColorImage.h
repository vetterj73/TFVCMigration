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

	COLORSTYLE GetColorStyle() {return _colorStyle;};
	void SetColorStyle(COLORSTYLE value);
	void SetChannelStoreSeperated(bool bValue);

	bool IsChannelStoredSeperate() {return _bChannelStoredSeperate;};

	bool Color2Luminance(Image* pGreyImg);

private:
	COLORSTYLE _colorStyle;
};

