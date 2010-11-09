/*-------------------------------------------------------------------------------
         Copyright © 2010 CyberOptics Corporation.  All rights reserved.
---------------------------------------------------------------------------------
    THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
    KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
    PURPOSE.

Module Name:

    PanelDescription.cpp

Abstract:

	PanelDescription.cpp provides the implementation to setup the CPanel object for
	2D SPI.

Environment:

    Managed C++ ('new' syntax)

---------------------------------------------------------------------------------
*/
#include "PanelDescription.h"

using namespace System;

namespace Cyber
{
	namespace SPIAPI
	{
		///
		// CPanel
		//

		//
		// Constructors
		//
		CPanel::CPanel(float lengthX, float lengthY )
		{
			_pPanel = new Panel();
			_pPanel->xLength(lengthX);
			_pPanel->yLength(lengthY);
		}

		CPanel::CPanel(PointD^ panelSize)
		{
			_pPanel = new Panel();
			_pPanel->xLength(panelSize->X);
			_pPanel->yLength(panelSize->Y);
		}

		CPanel::CPanel(System::Drawing::PointF panelSize)
		{
			_pPanel = new Panel();
			_pPanel->xLength(panelSize.X);
			_pPanel->yLength(panelSize.Y);
		}



		//
		// Methods
		//

		SPISTATUS CPanel::AddFeature(CFeature^ feature)
		{
			return (SPISTATUS)_pPanel->AddFeature((Feature*)(void*)feature->UnmanagedFeature);
		}

		void CPanel::ClearFeatures()
		{
			_pPanel->ClearFeatures();
		}

		void CPanel::RemoveFeature(int featureId)
		{
			_pPanel->RemoveFeature(featureId);
		}

		CFeature^ CPanel::GetFirstFeature()
		{
			if(_pPanel == NULL)
				return nullptr;

			Feature *pFeature = _pPanel->GetFirstFeature();
			if(pFeature == NULL)
				return nullptr;

			return ToManagedFeature(pFeature);
		}

		CFeature^ CPanel::GetNextFeature()
		{
			if(_pPanel == NULL)
				return nullptr;

			Feature *pFeature = _pPanel->GetNextFeature();
			if(pFeature == NULL)
				return nullptr;
			
			return ToManagedFeature(pFeature);
		}



		SPISTATUS CPanel::AddFiducial(CFeature^ fiducial)
		{
			return (SPISTATUS)_pPanel->AddFiducial((Feature*)(void*)fiducial->UnmanagedFeature);
		}

		void CPanel::ClearFiducials()
		{
			if(_pPanel == NULL)
				return;

			_pPanel->ClearFiducials();
		}

		void CPanel::RemoveFiducial(int fiducialId)
		{
			if(_pPanel == NULL)
				return;

			_pPanel->RemoveFiducial(fiducialId);
		}

		CFeature^ CPanel::GetFirstFiducial()
		{
			if(_pPanel == NULL)
				return nullptr;

			Feature *pFiducial = _pPanel->GetFirstFiducial();
			if(pFiducial == NULL)
				return nullptr;
			
			return ToManagedFeature(pFiducial);
		}

		CFeature^ CPanel::GetNextFiducial()
		{
			if(_pPanel == NULL)
				return nullptr;

			Feature *pFiducial = _pPanel->GetNextFiducial();
			if(pFiducial == NULL)
				return nullptr;
			
			return ToManagedFeature(pFiducial);
		}

		CFeature^ CPanel::ToManagedFeature(Feature *pFeature)
		{
			try
			{
				if(pFeature == NULL)
					return nullptr;
				
				switch(pFeature->GetShape())
				{
				case Feature::SHAPE_CROSS:
					{
						return gcnew CCross(pFeature);
					}
				case Feature::SHAPE_DIAMOND:
					{
						return gcnew CDiamond(pFeature);
					}
				case Feature::SHAPE_DISC:
					{
						return gcnew CDisc(pFeature);
					}
				case Feature::SHAPE_DONUT:
					{
						return gcnew CDonut(pFeature);
					}
				case Feature::SHAPE_RECTANGLE:
					{
						return gcnew CRectangle(pFeature);
					}
				case Feature::SHAPE_CYBER:
					{
						return gcnew CCyberShape(pFeature);
					}
				default:
					{
						return gcnew CFeature(pFeature);
					}
				}
			}
			catch(System::Exception^ e)
			{
				//throw;
				return nullptr;
			}

		}

		//
		// Properties
		//

		System::String^ CPanel::Name::get() 
		{ 
			return gcnew System::String(_pPanel->Name().c_str()); 
		}

		void CPanel::Name::set(System::String^ value) 
		{ 
			string name = (char*)(System::Runtime::InteropServices::Marshal::StringToHGlobalAnsi(value)).ToPointer();			
			_pPanel->Name(name); 
		}



		SPIAPI::PointD^ CPanel::PanelSize::get() 
		{ 
			SPIAPI::PointD^ p = gcnew PointD(_pPanel->xLength(), _pPanel->yLength());
			return p; 
		}

		void CPanel::PanelSize::set(SPIAPI::PointD^ value) 
		{ 
			_pPanel->xLength(value->X);
			_pPanel->yLength(value->Y);
		}

	} // Namespace
}
