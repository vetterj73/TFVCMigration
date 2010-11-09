// This is the main DLL file.

#include "stdafx.h"
#include "PanelConverter.h"
#include "Panel.h"
#include "Image16.h"
#include "Cad2Img.h"

namespace PanelToImage 
{
	void PanelConverter::ConvertPanel(System::IntPtr panel, double pixelSize, unsigned int columns, unsigned int rows,
		Byte* cadBuffer, Word* aperatureBuffer, bool drawCADROI)
	{
		Cad2Img c2i((Panel*)(void*)panel, columns, rows, cadBuffer, aperatureBuffer, pixelSize, drawCADROI);
	}
}