/*-------------------------------------------------------------------------------
         Copyright © 2010 CyberOptics Corporation.  All rights reserved.
---------------------------------------------------------------------------------
    THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
    KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
    PURPOSE.

Module Name:

    PanelDescription.h

Abstract:

	PanelDescription.h provides the API to setup the CPanel object for
	2D SPI.

Environment:

    Managed C++ ('new' syntax)

---------------------------------------------------------------------------------
*/
#pragma once


using namespace System::Collections::Generic;

#include "feature.h"
#include "CFeature.h"
#include "panel.h"

namespace Cyber
{
	namespace SPIAPI
	{
		/// \brief
		/// The CPanel class is used to communicate panel geometries for a 
		/// specific PCB panel (product) defined by a Gerber data file. This 
		/// includes panel size, feature shapes and locations, fiducial shapes 
		/// and locations, and ancillary panel attributes. The CPanel class is 
		/// passed to the SPI API through the CSPIAPI method ChangeProduct. 
		/// A CPanel class contains two lists: a feature list and a fiducial 
		/// list. 
		/// 
		/// \remarks Classes used to describe features and fiducials are described 
		/// in the Feature and derived Gerber Data Objects.  The shapes are SMEMA
		/// SRFF defined objects, with the exception of CyberShape, which is a 
		/// CyberOptics defined format.
		///
		/// \attention The \ref PanelSize property must be set large enough to
		/// contain all features and fiducials.
		///
		/// \image html CPanel-Properties.png CPanel properties
		///
		/// \par CPanel Properties
		/// The \ref CPanel class has properties illustrated in the figure 
		/// above which define the position and orientation of the CAD as 
		/// it relates the PCB panel on the conveyor.
		/// \n\n
		/// \b PanelOrigin is an enumeration that identifies which one of 
		/// the four corners of the PCB is the panel origin. The example 
		/// shown in the figure above is a \b PanelOrigin of \b RightBack. 
		/// The default \b PanelOrigin in \b LeftFront.
		/// \n\n
		/// \b CADOriginOffset defines the X and Y offset of the CAD 
		/// origin with respect to the panel origin. It is applied prior 
		/// to \ref CPanel rotation so will likely be positive 
		/// X and Y values.
		/// \n\n
		/// \b Rotation is the CAD rotation about the panel origin, 
		/// with a positive value representing a clockwise rotation.
		/// \n\n
		/// \b Size is the total PCB panel X and Y dimensions.
		/// \n\n
		/// \b PanelOffset is the delay distance from the nominal home 
		/// position. A negative \b HomeOffset value specifies a distance 
		/// proceeding the nominal home position and is limited by the 
		/// distance between the panel detect sensor and the nominal home 
		/// position. 
		public ref class CPanel
		{
		public:

			#pragma region enumerations

			#pragma endregion

			#pragma region Constructor(s)

			/// CPanel Constructor
			///
			/// \param panelSize A \ref PointD class containing the X and Y dimenstions
			/// of the panel.
			CPanel(double lengthX, double lengthY );
			CPanel(System::Drawing::PointF panelSize);

			#pragma endregion

			#pragma region Methods

			/// Adds a feature to the list of features to inspect.
			///
			/// \param feature A \ref CFeature object to inspect.
			int AddFeature(CFeature^ feature);

			///
			/// Removes all features from the panel's inspection list.
			void ClearFeatures();

			/// Remove a specific feature from the inspection list.
			///
			/// \param featureId The ID of the feature to erase.
			void RemoveFeature(int featureId);

			/// Adds a fiducial to the list of fiducials to use for inspeciton.
			///
			/// \param fiducial A \ref CFeature object to find.
			int AddFiducial(CFeature^ fiducial);

			///
			/// Removes all fiducials from the panel's list.
			void ClearFiducials();

			/// Remove a specific fiducial from the inspection list.
			///
			/// \param fiducialId The ID of the fiducial to erase.
			void RemoveFiducial(int fiducialId);

			#pragma region Properties

			/// 
			/// The name of the panel.
			property System::String^ Name
			{
				System::String^ get();
				void set(System::String^ value);
			}

			/// The total PCB panel X dimension.
			/// in meters.
			property double PanelSizeX
			{
				double get();
				void set(double value);
			}


			/// The total PCB panel Y dimension.
			/// in meters.
			property double PanelSizeY
			{
				double get();
				void set(double value);
			}


			/// \internal
			property System::IntPtr UnmanagedPanel
			{
				System::IntPtr get() { return safe_cast<System::IntPtr>(_pPanel); }
			}

			/// Get the number of features in the CPanel object.
			property int NumberOfFeatures
			{
				int get()
				{
					if(_pPanel == NULL)
						return 0;
					return _pPanel->NumberOfFeatures();
				}
			}

			/// Get the number of fiducials in the CPanel object.
			property int NumberOfFiducials
			{
				int get()
				{
					if(_pPanel == NULL)
						return 0;
					return _pPanel->NumberOfFiducials();
				}
			}

			/// Get the first feature from the feature list defined in the 
			/// CPanel object.  This method must called first when iterating
			/// through the feature list.
			///
			/// \returns A pointer to the first feature or \b NULL if the
			/// feature list is empty.
			///
			/// \sa GetNextFeature GetFirstFiducial GetNextFiducial
			CFeature^ GetFirstFeature();

			/// Get the next feature from the feature list defined in the 
			/// CPanel object.  \b GetFirstFeature method must be called
			/// before \b GetNextFeature when iterating through the 
			/// feature list.
			///
			/// \returns A pointer to the next feature or \b NULL if end
			/// of the feature list has been reached.
			///
			/// \sa GetFirstFeature GetFirstFiducial GetNextFiducial
			CFeature^ GetNextFeature();

			/// Get the first fiducial from the ficucial list defined in the 
			/// CPanel object.  This method must called first when iterating
			/// through the fiducial list.
			///
			/// \returns A pointer to the first fiducial or \b NULL if the
			/// fiducial list is empty.
			///
			/// \sa GetFirstFeature GetNextFeature GetNextFiducial
			CFeature^ GetFirstFiducial();

			/// Get the next fiducial from the ficucial list defined in the 
			/// CPanel object.  \b GetFirstFiducial method must be called
			/// before \b GetNextFiducial when iterating through the 
			/// fiducial list.
			///
			/// \returns A pointer to the next fiducial or \b NULL if end
			/// of the fiducial list has been reached.
			///
			/// \sa GetFirstFeature GetNextFeature GetFirstFiducial
			CFeature^ GetNextFiducial();

			#pragma endregion

		protected:

			Panel *_pPanel;

		private:
			CFeature^ ToManagedFeature(Feature* pFeature);
		};
	}
}
