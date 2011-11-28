/*-------------------------------------------------------------------------------
         Copyright © 2010 CyberOptics Corporation.  All rights reserved.
---------------------------------------------------------------------------------
    THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
    KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
    PURPOSE.

Module Name:

    FeatureDescription.h

Abstract:

	FeatureDescription.h provides the API to setup feature objects for
	2D SPI.

Environment:

    Managed C++ ('new' syntax)

---------------------------------------------------------------------------------
*/
#pragma once

#include "feature.h"

using namespace System::Collections::Generic;

namespace Cyber
{
	namespace MPanel
	{
		/// \brief
		/// The \b CFeature class is the base class for all feature and fiducial 
		/// shape classes. It contains the feature or fiducial location and 
		/// rotation in the panel coordinate system as well as a feature 
		/// reference ID used for reporting results.
		public ref class CFeature
		{
		public:
			#pragma region Enumerations

			/// Shape definition of each Feature
			enum class ShapeType
			{
				Undefined=-1,
				Cross,
				Diamond,
				Disc,
				Donut,
				Rectangle,
				Triangle,
				CheckerPattern,
				CyberShape
			};

			#pragma endregion

			/// \internal
			/// Only derived classes should create CFeature.
			CFeature(Feature *pFeature)
			{
				_pFeature = pFeature;
			}

			#pragma region Properties

			/// 
			/// The shape of the Feature as defined by \ref ShapeType
			property ShapeType Type
			{
				ShapeType get() { return (ShapeType)_pFeature->GetShape(); }
			}
 
			/// ReferenceID is used to identify a specific feature for reporting results. It must be
			/// unique for each feature within a panel description. The suggested value is the feature
			/// index in the panel list of features (see CPanel Class).
			property int ReferenceID
			{
				int get() { return _pFeature->GetId(); }
			}

			/// 
			/// Position in X from the panel origin in meters.
			property double PositionX
			{
				double get() { return _pFeature->GetCadX(); }
			}

			/// 
			/// Position in Y from panel origin in meters.
			property double PositionY
			{
				double get() { return _pFeature->GetCadY(); }
			}

			/// 
			/// Rotation about the feature origin in degrees.
			property double Rotation
			{
				double get() { return _pFeature->GetRotation(); }
			}

			/// \internal
			property System::IntPtr UnmanagedFeature
			{
				System::IntPtr get() { return safe_cast<System::IntPtr>(_pFeature); }
			}

			/// \internal
			property double InspectionLeft
			{
				double get() { return _pFeature->GetInspectionArea().p1.x; }
			}

			/// \internal
			property double InspectionTop
			{
				double get() { return _pFeature->GetInspectionArea().p1.y; }
			}

			/// \internal
			property double InspectionRight
			{
				double get() { return _pFeature->GetInspectionArea().p2.x; }
			}

			/// \internal
			property double InspectionBottom
			{
				double get() { return _pFeature->GetInspectionArea().p2.y; }
			}

			/// \internal
			property unsigned int AperatureValue
			{
				unsigned int get() { return _pFeature->GetApertureValue(); }
			}

			/// \internal
			property double NominalArea
			{
				double get() { return _pFeature->GetNominalArea(); }
			}

 			#pragma endregion

		protected:
			#pragma region Constructor(s)

			/// \internal
			/// Only derived classes should create CFeature.
			CFeature();

			#pragma endregion

			Feature *_pFeature;
		};



		/// \brief
		/// The \b CCross class defines the two dimensional geometry of a cross 
		/// shape. It is used to define a feature or a fiducial.
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. See the SMEMA SRFF 
		/// Specification for a graphical representation of the cross and its properties.
		public ref class CCross : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CCross Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param rotation The rotation about the feature origin.
			/// \param sizeX The width of the base of the cross.
			/// \param sizeY The height of the cross.
			/// \param legSizeX The width of the base leg of the cross.
			/// \param legSizeY The width of the height leg of the cross.
			CCross(int referenceID, double positionX, double positionY, double rotation,
				double sizeX, double sizeY, double legSizeX, double legSizeY );

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CCross(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			#pragma region Properties

			/// 
			/// The width of the base of the cross.
			property double SizeX
			{
				double get() { return ((CrossFeature*)_pFeature)->GetSizeX(); }
			}

			/// 
			/// The height of the cross.
			property double SizeY
			{
				double get() { return ((CrossFeature*)_pFeature)->GetSizeY(); }
			}

			/// 
			/// The width of the base leg of the cross.
			property double LegSizeX
			{
				double get() { return ((CrossFeature*)_pFeature)->GetLegSizeX(); }
			}

			/// 
			/// The width of the height leg of the cross.
			property double LegSizeY
			{
				double get() { return ((CrossFeature*)_pFeature)->GetLegSizeY(); }
			}

#pragma endregion

		};



		/// \brief
		/// The \b CDiamond class defines the two dimensional geometry of a diamond shape. 
		/// It is used to define a feature or a fiducial.
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. See the SMEMA SRFF 
		/// Specification for a graphical representation of the diamond and its properties.
		public ref class CDiamond : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CDiamond Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param rotation The rotation about the feature origin.
			/// \param sizeX The width of the base of the diamond.
			/// \param sizeY The height of the diamond.
			CDiamond(int referenceID, double positionX, double positionY, double rotation,
				double sizeX, double sizeY );

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CDiamond(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			#pragma region Properties
			///
			/// The width of the base of the diamond.
			property double SizeX
			{
				double get() { return ((DiamondFeature*)_pFeature)->GetSizeX(); }
			}

			///
			/// The height of the diamond.
			property double SizeY
			{
				double get() { return ((DiamondFeature*)_pFeature)->GetSizeY(); }
			}

			#pragma endregion
		};



		/// \brief
		/// The \b CDisc class defines the two dimensional geometry of a disc shape. 
		/// It is used to define a feature or a fiducial.
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. See the SMEMA SRFF 
		/// Specification for a graphical representation of the disc and its properties.
		public ref class CDisc : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CDisc Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param diameter The diameter of the disc.
			CDisc(int referenceID, double positionX, double positionY, double diameter);

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CDisc(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			#pragma region Properties
			///
			/// The diameter of the disc.
			property double Diameter
			{
				double get() { return ((DiscFeature*)_pFeature)->GetDiameter(); }
			}
			#pragma endregion
		};



		/// \brief
		/// The \b CDonut class defines the two dimensional geometry of a donut shape. 
		/// It is used to define a feature or a fiducial.
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. See the SMEMA SRFF 
		/// Specification for a graphical representation of the donut and its properties.
		public ref class CDonut : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CDonut Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param diameterInside The inner diameter of the donut.
			/// \param diameterOutside The outer diameter of the donut.
			CDonut(int referenceID, double positionX, double positionY, double diameterInside, double diameterOutside );

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CDonut(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			#pragma region Properties
			///
			/// The inner diameter of the donut.
			property double DiameterInside
			{
				double get() { return ((DonutFeature*)_pFeature)->GetDiameterInside(); }
			}

			///
			/// The outer diameter of the donut.
			property double DiameterOutside
			{
				double get() { return ((DonutFeature*)_pFeature)->GetDiameterOutside(); }
			}
			#pragma endregion
		};



		/// \brief
		/// The \b CRectangle class defines the two dimensional geometry of a rectangle shape. 
        /// It is used to define a feature or a fiducial.
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. See the SMEMA SRFF 
		/// Specification for a graphical representation of the rectangle and its properties.
		public ref class CRectangle : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CRectangle Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param rotation The rotation about the feature origin.
			/// \param sizeX The width of the base of the rectangle.
			/// \param sizeY The length of the rectangle.
			/// \param sizeZ The height of the rectangle.
			CRectangle(int referenceID, double positionX, double positionY, double rotation,
				double sizeX, double sizeY, double sizeZ);

			CRectangle(int referenceID, double positionX, double positionY, double rotation,
				double sizeX, double sizeY);

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CRectangle(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			#pragma region Properties
			///
			/// The width of the base of the rectangle.
			property double SizeX
			{
				double get() { return ((RectangularFeature*)_pFeature)->GetSizeX(); }
			}

			///
			/// The height of the rectangle.
			property double SizeY
			{
				double get() { return ((RectangularFeature*)_pFeature)->GetSizeY(); }
			}
			#pragma endregion
		};



		/// \brief
		/// The \b CTriangle class defines the two dimensional geometry of a triangle shape. 
		/// It is used to define a feature or a fiducial.
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. See the SMEMA SRFF 
		/// Specification for a graphical representation of the triangle and its properties.
		public ref class CTriangle : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CTriangle Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param rotation The rotation about the feature origin.
			/// \param sizeX The width of the base of the triangle.
			/// \param sizeY The height of the triangle.
			/// \param offsetX The distance along the X-axis from the left of the base of the triangle to the tip.
			CTriangle(int referenceID, double positionX, double positionY, double rotation,
				double sizeX, double sizeY, double offsetX );

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CTriangle(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			#pragma region Properties
			///
			/// The width of the base of the triangle.
			property double SizeX
			{
				double get() { return ((TriangleFeature*)_pFeature)->GetSizeX(); }
			}

			///
			/// The height of the triangle.
			property double SizeY
			{
				double get() { return ((TriangleFeature*)_pFeature)->GetSizeY(); }
			}

			///
			/// The distance along the X-axis from the left of the base of the triangle to the tip.
			property double OffsetX
			{
				double get() { return ((TriangleFeature*)_pFeature)->GetOffset(); }
			}

			#pragma endregion
		};


		/// \brief
		/// The \b CCheckerPattern class defines the two dimensional geometry of a CCheckerPattern shape. 
		// CAD polygon points in CW order (two squares with the same size)
		//		    *-------* 		--------------
		//          |		|               h
		//          |       |               e
		//          *-------*-------*       i
		//					|       |       g
		//					|		|       h
		//				    *-------* ------t------
		//			|  size			|
		/// It is used to define a feature or a fiducial.
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. 
		/// Specification for a graphical representation of the CheckerPattern shape and its properties.
		public ref class CCheckerPattern : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CCheckerPattern Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param rotation The rotation about the feature origin.
			/// \param sizeX The width of the base of the CheckerPattern shape (the "size" shows in the figure) .
			/// \param sizeY The height of the CheckerPattern shape.
			CCheckerPattern(int referenceID, double positionX, double positionY, double rotation,
				double sizeX, double sizeY );

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CCheckerPattern(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			#pragma region Properties
			///
			/// The width of the base of the CheckerPattern.
			property double SizeX
			{
				double get() { return ((CheckerPatternFeature*)_pFeature)->GetSizeX(); }
			}

			///
			/// The height of the CheckerPattern.
			property double SizeY
			{
				double get() { return ((CheckerPatternFeature*)_pFeature)->GetSizeY(); }
			}
			#pragma endregion
		};

		/// \brief
		/// The \b CSegment class defines one segment of a list of segments used to define the two 
		/// dimensional geometry of a CCyberShape. All segments are circular arcs or line segments.
		/// 
		/// \remarks The CSegment positions are defined relative to the origin of the \b CCyberShape. 
		/// The final segment is included to define the endpoint of the previous segment. The actual 
		/// segment will be ignored. It is a good practice to use a duplicate of the first segment 
		/// with PenDown set to false.
		public ref class CCyberSegment
		{
		public:

			#pragma region Constructor(s)

			/// \b CCyberSegment Constructor
			/// 
			/// \param line A boolean indicating whether the segment is a line segment or an arc.
			/// \param penDown A boolean indicating whether the segment is used to draw or move the pen.
			/// \param clockwiseArc A boolean used for arc segments that defines whether the pen is to move
			/// in a clockwise or counter-clockwise direction around the circle center to the endpoint.
			/// \param positionX The X position of the ending point for the segment.  See note.
			/// \param positionY The Y position of the ending point for the segment.  See note.
			/// \param arcX The X position of the center of the circle when drawing an arc segment.
			/// \param arcY The Y position of the center of the circle when drawing an arc segment.
			///
			/// \note The \b positionX and \b positionY refer to the ending point of the segment, except
			/// for the first segment of a CyberShape.  The first segment should be a line segment with 
			/// the position defining the starting point of the CyberShape.  \sa CCyberShape
			CCyberSegment(bool line, bool penDown, bool clockwiseArc, double positionX,
				double positionY, double arcX, double arcY);

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CCyberSegment(CyberSegment *pCyberSegment)
			{
				_pCyberSegment = pCyberSegment;
			}

			#pragma endregion

			#pragma region Properties
			///
			/// A boolean indicating whether the segment is a line segment or an arc.
			property bool Line
			{
				bool get() { return _pCyberSegment->GetLine(); }
			}

			///
			/// A boolean indicating whether the segment is used to draw or move the pen.
			property bool PenDown
			{
				bool get() { return _pCyberSegment->GetPenDown(); }
			}

			///
			/// A boolean used for arc segments that defines whether the pen is to move 
			/// in a clockwise or counter-clockwise direction around the circle center to the endpoint.
			property bool ClockwiseArc
			{
				bool get() { return _pCyberSegment->GetClockwiseArc(); }
			}

			///
			/// The X position of the starting point for the segment.
			property double PositionX
			{
				double get() { return _pCyberSegment->GetPositionX(); }
			}

			///
			/// The Y position of the starting point for the segment.
			property double PositionY
			{
				double get() { return _pCyberSegment->GetPositionY(); }
			}

			///
			/// The X position of the center of the circle when drawing an arc segment.
			property double ArcX
			{
				double get() { return _pCyberSegment->GetArcX(); }
			}

			///
			/// The Y position of the center of the circle when drawing an arc segment.
			property double ArcY
			{
				double get() { return _pCyberSegment->GetArcY(); }
			}
			
			/// \internal
			property System::IntPtr UnmanagedSegment
			{
				System::IntPtr get(){return safe_cast<System::IntPtr>(_pCyberSegment);}
			}
			#pragma endregion

		protected:
			CyberSegment *_pCyberSegment;
		};



		/// \brief
		/// The \b CCyberShape class defines the two dimensional geometry of 
		/// any shape that is not one of the standard shape classes defined above.
		///
		/// \b Notes:
		/// \li The final end point is explicitly defined. So the final 
		/// CyberSegment x and y position will be equal to the first x and y 
		/// position to close the shape.
		/// \li Shapes are placed relative to the x,y,z position of their parent 
		/// Location. The shape is placed around that Location point of origin 
		/// in the same manner that current shapes are placed.
		/// \li The arc positions x and y (ArcX, ArcY) describe the center of 
		/// the circle on a line perpendicular to the midpoint of the start 
		/// point and endpoint. The radius can be easily determined as the 
		/// distance from the center to either start or end point. The same 
		/// distance is held by all points between along the arc. The Boolean 
		/// clockwise Arc determines if the arc is drawn ‘north’ or ‘south’ of 
		/// the start and end point.  
		/// \li Arc drawing and all arc arguments are only relevant when the 
		/// Boolean bLine is false. In the case of a line definition, arc 
		/// arguments may be a defined as a ‘*’ for clarity of intent. In the 
		/// case of an arc definition, all arguments need to be defined 
		/// according to type.
		/// \li The vertices of the shape description shall be ordered in a 
		/// clockwise manner.  Counter-clockwise ordered vertices should be 
		/// reserved to describe cutouts to shapes (e.g. the shape of the 
		/// interior hole of a donut shape).  It should be recognized that 
		/// solder paste apertures cannot be defined using these type of  
		/// “floating” cutouts that are defined with counter-clockwise order 
		/// vertices.  Since some CAD software has defined CCW order vertices 
		/// this way, this requirement is added so that future changes to 
		/// implementation don’t rely on counterclockwise defined pads.  
		///
		/// \remarks The constructor parameters referenceID, X, Y and rotation 
		/// initialize the base class \b CFeature. See the CyberOptics document
		/// \a Odd \a Shaped \a Pads for a description of the format used to 
		/// describe a CyberShape feature or fiducial.
		public ref class CCyberShape : public CFeature
		{
		public:

			#pragma region Constructor(s)

			/// \b CCyberShape Constructor
			/// 
			/// \param referenceID The unique reference ID for this feature.
			/// \param positionX The position in X from the panel origin.
			/// \param positionY The position in Y from the panel origin.
			/// \param rotation The rotation about the feature origin.
			CCyberShape(int referenceID, double positionX, double positionY, double rotation);

			/// \internal
			/// Needed for constructing Managed class from Unmanaged pointer
			CCyberShape(Feature *pFeature) : CFeature(pFeature) {} ;

			#pragma endregion

			/// \b AddSegment addes a \b CyberSegment to a list of segment classes
			/// used to define the non standard shape.
			void AddSegment(CCyberSegment^ segment);

			/// Get the number of fiducials in the CPanel object.
			int GetNumberSegments();

			/// Get the first CCyberSegment from the segment list defined in the 
			/// CCyberShape object.  This method must called first when iterating
			/// through the CCyberSegment list.
			///
			/// \returns A pointer to the first CCyberSegment or \b NULL if the
			/// CCyberSegment list is empty.
			///
			/// \sa GetNextSegment
			CCyberSegment^ GetFirstSegment();

			/// Get the next CCyberSegment from the CCyberSegment list defined in the 
			/// CCyberShape object.  \b GetFirstSegment method must be called
			/// before \b GetNextSegment when iterating through the 
			/// feature list.
			///
			/// \returns A pointer to the next CCyberSegment or \b NULL if end
			/// of the CCyberSegment list has been reached.
			///
			/// \sa GetFirstSegment
			CCyberSegment^ GetNextSegment();
		};


	} // SPIAPI namespace
} // Cyber namespace
