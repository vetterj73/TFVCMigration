#include "StdAfx.h"
#include "PanelCreator.h"
#include "PanelConverter.h"

using namespace System;

namespace PanelToImage
{
	PanelCreator::PanelCreator(System::IntPtr panel, PanelImages panelImages, double pixelSize, bool drawCADROI) :
			_cadImage(0),
			_aptImage(0)
	{
		Panel* pPanel = (Panel*)(void*)panel;

		// If CAD Image is specified, create the cad image
		if((int)panelImages & (int)PanelImages::CAD)
		{
			_cadImage = new Image();
			_cadImage->ConfigureFromPixels(pPanel->xLength(), pPanel->yLength(), 
				pixelSize, pixelSize);

			Cad2Img c2i((Panel*)(void*)panel, _cadImage->Columns(), _cadImage->Rows(), 
				_cadImage->GetBuffer(), 0, pixelSize, drawCADROI);
		}

		// If Aperture Image is specfied, create the aperture image
		if((int)panelImages & (int)PanelImages::Aperture)
		{
			_aptImage = new Image16();
			_aptImage->ConfigureFromPixels(pPanel->xLength(), pPanel->yLength(),
				pixelSize, pixelSize);

			Cad2Img c2i((Panel*)(void*)panel, _aptImage->Columns(), _aptImage->Rows(), 
				0, _aptImage->GetBuffer16(), pixelSize, drawCADROI);
		}
	}

	PanelCreator::~PanelCreator(void)
	{
		FreeCADBuffer();
		FreeApertureBuffer();
	}


	void PanelCreator::FreeCADBuffer()
	{
		if(_cadImage)
		{
			delete _cadImage;
			_cadImage = 0;
		}
	}

	System::IntPtr PanelCreator::GetCADBuffer()
	{
		if(_cadImage)
			return System::IntPtr(_cadImage->GetBuffer());
		else
			return System::IntPtr();
	}


	void PanelCreator::FreeApertureBuffer()
	{
		if(_aptImage)
		{
			delete _aptImage;
			_aptImage = 0;
		}
	}

	System::IntPtr PanelCreator::GetApertureBuffer()
	{
		if(_aptImage)
			return System::IntPtr(_aptImage->GetBuffer16());
		else
			return System::IntPtr();
	}


	unsigned int PanelCreator::Columns::get()
	{
		if(_cadImage)
			return _cadImage->Columns();
		else if(_aptImage)
			return _aptImage->Columns();
		else
			return 0;
	}

	unsigned int PanelCreator::Rows::get()
	{
		if(_cadImage)
			return _cadImage->Rows();
		else if(_aptImage)
			return _aptImage->Rows();
		else
			return 0;
	}


	unsigned int PanelCreator::RowStrideBytes::get()
	{
		if(_cadImage)
			return _cadImage->ByteRowStride();
		else if(_aptImage)
			return _aptImage->ByteRowStride();
		else
			return 0;
	}


} // end namespace PanelToImage