/*

	Converts pad CAD into a rasterized image

	Input is a Panel object and pixel size

	Output is an CadImage object

*/

#pragma once

#include "Panel.h"
#include "Image16.h"

/// \brief
/// The Cad@Img class is a static class that is used to convert a CAD representation of an panel
/// to an image representation of a panel.
///
class Cad2Img
{
public:
	///
	///	DrawCAD - Draws a Panel to an 8 bit image pointer 
	///
	static bool DrawCAD(Panel* pPanel, unsigned char* cadBuffer, bool DrawCADROI=false); 

	
	/// Draw 8bit height image to represent the (component) height of panel surface
	/// dHeightResolution: the height represented by each grey level
	static bool DrawHeightImage(Panel* pPanel, unsigned char* cadBuffer, double dHeightResolution);

	///
	///	DrawAperatures - Draws a Panel to a 16 bit image pointer (for PadStats)
	///
	static bool DrawAperatures(Panel* pPanel, unsigned short* aperatureBuffer, bool DrawCADROI = true); 

	///
	///	DrawMask - Draws cad with scaling (used to mask areas of images that may be cad).
	///
	static bool DrawMask(Panel* pPanel,	unsigned short* maskBuffer, double scale); 
};