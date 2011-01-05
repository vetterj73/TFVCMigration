// This is the main DLL file.

#include "PanelConverter.h"
#include "Panel.h"
#include "Image16.h"
#include "ManagedImage.h"
#include "Cad2Img.h"

namespace Cyber
{	namespace MPanel
	{
		void PanelConverter::ConvertPanel(System::IntPtr panel, double pixelSize, 
			unsigned int columns, unsigned int rows, unsigned int stride,
			System::IntPtr cadBuffer, System::IntPtr aperatureBuffer, bool drawCADROI)
		{
			Cad2Img c2i((Panel*)(void*)panel, columns, rows, stride, (unsigned char*)(void*)cadBuffer, (Word*)(void*)aperatureBuffer, pixelSize, drawCADROI);
		}
	}
}