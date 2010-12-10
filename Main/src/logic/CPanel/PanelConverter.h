// PanelConverter.h

#pragma once
#include "Image.h"
class Panel;

namespace Cyber
{
	namespace MPanel
	{
		public ref class PanelConverter
		{
			public:	
				static void ConvertPanel(
					System::IntPtr panel, double pixelSize, 
					unsigned int columns, unsigned int rows, unsigned int stride, 
					System::IntPtr cadBuffer, System::IntPtr aperatureBuffer, bool drawCADROI);

			protected:
				PanelConverter(){};
		};
	}
}
