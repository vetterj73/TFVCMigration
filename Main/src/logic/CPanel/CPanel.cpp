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
#include "CPanel.h"

using namespace System;

namespace Cyber
{
	namespace MPanel
	{
		///
		// CPanel
		//

		//
		// Constructors
		//
		CPanel::CPanel(double lengthX, double lengthY )
		{
			_pPanel = new Panel();
			_pPanel->xLength(lengthX);
			_pPanel->yLength(lengthY);
		}

		//
		// Methods
		//

		int CPanel::AddFeature(CFeature^ feature)
		{
			return _pPanel->AddFeature((Feature*)(void*)feature->UnmanagedFeature);
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

		int CPanel::AddFiducial(CFeature^ fiducial)
		{
			return _pPanel->AddFiducial((Feature*)(void*)fiducial->UnmanagedFeature);
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
			catch(System::Exception^)
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



		double CPanel::PanelSizeX::get() 
		{ 
			return _pPanel->xLength();
		}

		void CPanel::PanelSizeX::set(double X) 
		{ 
			_pPanel->xLength(X);
		}

		double CPanel::PanelSizeY::get() 
		{ 
			return _pPanel->yLength();
		}

		void CPanel::PanelSizeY::set(double Y) 
		{ 
			_pPanel->yLength(Y);
		}

	} // Namespace
}
