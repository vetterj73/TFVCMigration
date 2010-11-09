#pragma once

#include "Cad2Img.h"

namespace PanelToImage
{

	public enum class PanelImages
	{
		Aperture  = 1,
		CAD       = 2,
		AllImages = 3,
	};

	public ref class PanelCreator
	{
	public:

		PanelCreator(System::IntPtr panel, PanelImages panelImages, double pixelSize, bool drawCADROI);
		~PanelCreator(void);

		void              FreeCADBuffer();
		System::IntPtr    GetCADBuffer();

		void              FreeApertureBuffer();
		System::IntPtr    GetApertureBuffer();

		property unsigned int Columns
		{
			unsigned int get();
		}

		property unsigned int Rows
		{
			unsigned int get();
		}

		property unsigned int RowStrideBytes
		{
			unsigned int get();
		}


	private:

		// Pointers to unmanged Image objects
		Image*   _cadImage;
		Image16* _aptImage;
	};
} // end namespace PanelToImage