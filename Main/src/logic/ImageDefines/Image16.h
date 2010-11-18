#pragma once

#include "Image.h"

/*
	Generic 16 bit image
*/
class Image16 : public Image
{
	public:
		Image16();

	// Helper Functions for 16 bit access - this is an override from Image...
	unsigned short* GetBuffer16();
	Byte* PtrToPixel(unsigned int column, unsigned int row);
	virtual void SetPixelValue(unsigned int index, unsigned int value);
};

