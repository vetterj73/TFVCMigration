#include "Image16.h"


Image16::Image16():
Image(2)
{
}

unsigned short* Image16::GetBuffer16(){return (unsigned short*)_buffer;};

unsigned short* Image16::GetBuffer16(unsigned int column, unsigned int row)
{
	if(_buffer == NULL)
		return NULL;
	return &(GetBuffer16()[row*PixelRowStride()+column]);
}

void Image16::SetPixelValue(unsigned int index, unsigned int value)
{
	if(index >= BufferSizeInPixels())
		return;
	((unsigned short*)_buffer)[index] = value;
}
