/*

	Converts pad CAD into a rasterized image

	Input is a Panel object and pixel size

	Output is an CadImage object

*/

#pragma once

#include "Panel.h"
#include "Image16.h"

class Cad2Img
{
public:

	Cad2Img(	Panel*			p,
				unsigned int columns,
				unsigned int rows,
				unsigned int stride,
				unsigned char* cadBuffer,
				Word * aptBuffer,
				double resolution,
				bool DrawCADROI = true);

	static bool DrawCAD(Panel* pPanel, unsigned char* cadBuffer, bool DrawCADROI=false); 

	static bool DrawMask(Panel* pPanel,	unsigned short* cadBuffer, double scale); 

	static bool DrawAperatures(Panel* pPanel, unsigned short* cadBuffer, bool DrawCADROI = true); 

private:

	void			DrawPads(bool drawCADROI);
	bool            _drawApt;
	bool            _drawCad;
	Panel*			_pPanel;
	Image			_cadImage;
	Image16	        _aptImage;
	double			_resolution;

	// draw the image as columns correspond with x, 
	// rows correspond with y
	bool _rightSideUp;
};