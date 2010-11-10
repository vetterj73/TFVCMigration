// PanelConverter.h

#pragma once

#include "typedefs.h"
class Panel;

namespace PanelToImage 
{
	public ref class PanelConverter
	{
		public:	
			static void ConvertPanel(
				System::IntPtr panel, double pixelSize, unsigned int columns, unsigned int rows,
				Byte* cadBuffer, Word* aperatureBuffer, bool drawCADROI);

		protected:
			PanelConverter(){};
	};
}