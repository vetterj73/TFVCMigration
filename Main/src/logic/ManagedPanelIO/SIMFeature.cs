////////////////////////////////////////////////////////////////////////////////
// 
// Filename: SIMFeature.cs
// 
// Modification History:
// Name: Martin Lascelles
// Date: 07/11/2009
// 
// 
////////////////////////////////////////////////////////////////////////////////
// This software program ('The Software') is the proprietary product of DEK
// International GmbH. and is protected by copyright laws and international
// treaty. You must treat the software like any other copyrighted material,
// except that you may make one copy of the software solely for backup or
// archival purposes. Copyright laws prohibit making additional copies of the
// software for any other reasons. You may not copy the written materials
// accompanying the software.
// Copyright (c) 2009  DEK International GmbH., All Rights Reserved
// 
// Code in this file conforms to the standards described in the
// Coding Standards Document, Specification DSS111.
////////////////////////////////////////////////////////////////////////////////
using System;
//using System.Text;
using System.Runtime.Serialization;
using System.Collections;
using System.Drawing;

namespace MPanelIO
{
	////////////////////////////////////////////////////////////////////////
	/// <summary>
	/// Name: t_FeatureType
	/// WARNING: this must the corresponding enum in the CSPIAPI module and in
	/// ISIMInterface.cs which cannot reference this assembly.
	/// </summary>
	//////////////////////////////////////////////////////////////////////
	public enum t_FeatureType
	{
		e_Cross = 0,
		e_Diamond = 1,
		e_Disc = 2,
		e_Donut = 3,
		e_Rectangle = 4,
		e_Triangle = 5,
		e_CyberShape = 6
	}
	//////////////////////////////////////////////////////////////////////////////
	/// <summary>
	/// Name: CSIMFeature class
	/// Description: Base class for features that the SIM can inspect. Serializable so 
	/// can be sent to remote SIM.
	/// </summary>
	//////////////////////////////////////////////////////////////////////////////
	[Serializable]
	[System.Xml.Serialization.XmlInclude( typeof( CSIMShape ) )]
	[System.Xml.Serialization.XmlInclude( typeof( CSIMDisc ) )]
	[System.Xml.Serialization.XmlInclude( typeof( CSIMRectangle ) )]
	public class CSIMFeature : ISerializable
	{
		#region Attribute and corresponding properties
		protected t_FeatureType m_Type;
		private float m_PositionX;
		private float m_PositionY;
		private int m_ReferenceID;
		private float m_Rotation;

		public float PositionX { get { return m_PositionX; } set { m_PositionX = value; } }
		public float PositionY { get { return m_PositionY; } set { m_PositionY = value; } }
		public int ReferenceID { get { return m_ReferenceID; } set { m_ReferenceID = value; } }
		public float Rotation { get { return m_Rotation; } set { m_Rotation = value; } }
		public t_FeatureType Type { get { return m_Type; } set { m_Type = value; } }
		#endregion Attribute and corresponding properties

		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMFeature 
		/// Description: Constructor
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMFeature( int theReferenceID, float thePositionX, float thePositionY, float theRotation )
		{
			m_PositionX = thePositionX;
			m_PositionY = thePositionY;
			m_ReferenceID = theReferenceID;
			m_Rotation = theRotation;
		}
		public CSIMFeature(){}

		public RectangleF OutLineRect
		{
			get
			{
				return GetOutLineRect();
			}
		}

		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetOutlineRect
		/// Description: Get outline (bounding) rectangle of Feature
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		protected virtual RectangleF GetOutLineRect()
		{
			return new RectangleF( m_PositionX, m_PositionY, 0, 0 );
		}
		#region ISerializable Members
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMFeature 
		/// Description: Deserialization constructor.
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMFeature( SerializationInfo theInfo, StreamingContext theStreamingContext )
		{
			//Get the values from info and assign them to the appropriate properties
			PositionX = ( float )theInfo.GetValue( "PositionX", typeof( float ) );
			PositionY = ( float )theInfo.GetValue( "PositionY", typeof( float ) );
			ReferenceID = ( int )theInfo.GetValue( "ReferenceID", typeof( int ) );
			Rotation = ( float )theInfo.GetValue( "Rotation", typeof( float ) );
			Type = ( t_FeatureType )theInfo.GetValue( "Type", typeof( t_FeatureType ) );
		}

		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetObjectData 
		/// Description: Serialization function
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public virtual void GetObjectData( SerializationInfo theInfo, StreamingContext theStreamingContext )
		{
			theInfo.AddValue( "PositionX", PositionX );
			theInfo.AddValue( "PositionY", PositionY );
			theInfo.AddValue( "ReferenceID", ReferenceID );
			theInfo.AddValue( "Rotation", Rotation );
			theInfo.AddValue( "Type", Type );
		}
		#endregion
	}
	//////////////////////////////////////////////////////////////////////////////
	/// <summary>
	/// Name: CSIMFeature class
	/// Description: Base class for features that the SIM can inspect. Serializable so 
	/// can be sent to remote SIM.
	/// </summary>
	//////////////////////////////////////////////////////////////////////////////
	[Serializable]
	public class CSIMDisc : CSIMFeature
	{
		public CSIMDisc( int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theDiameter )
			: base( theReferenceID, thePositionX, thePositionY, theRotation )
		{
			m_Type = t_FeatureType.e_Disc;
			m_Diameter = theDiameter;
		}
		private float m_Diameter;
		public float Diameter { get { return m_Diameter; } set { m_Diameter = value; } }

		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetOutlineRect
		/// Description: Get outline (bounding) rectangle of Feature
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		protected override RectangleF GetOutLineRect()
		{
			return new RectangleF( PositionX, PositionY, m_Diameter, m_Diameter );
		}
		public CSIMDisc(){}

		#region ISerializable Members
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMDisc 
		/// Description: Deserialization constructor.
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMDisc( SerializationInfo theInfo, StreamingContext theStreamingContext )
			: base( theInfo, theStreamingContext )
		{
			//Get the values from info and assign them to the appropriate properties
			Diameter = ( float )theInfo.GetValue( "Diameter", typeof( float ) );
		}

		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetObjectData
		/// Description: Serialization function
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public override void GetObjectData( SerializationInfo theInfo, StreamingContext theStreamingContext )
		{
			base.GetObjectData( theInfo, theStreamingContext );
			theInfo.AddValue( "Diameter", Diameter );
		}
		#endregion
	}
	[Serializable]
	public class CSIMRectangle : CSIMFeature
	{
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMRectangle 
		/// Description: Constructor
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMRectangle( int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSizeX, float theSizeY, float theSizeZ=0 )
			: base( theReferenceID, thePositionX, thePositionY, theRotation )
		{
			m_Type = t_FeatureType.e_Rectangle;
			m_SizeX = theSizeX;
			m_SizeY = theSizeY;
            m_SizeZ = theSizeZ;
		}
		private float m_SizeX;
		private float m_SizeY;
        private float m_SizeZ;
		public float SizeX { get { return m_SizeX; } set { m_SizeX = value; } }
		public float SizeY { get { return m_SizeY; } set { m_SizeY = value; } }
        public float SizeZ { get { return m_SizeZ; } set { m_SizeZ = value; } }

		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetOutlineRect
		/// Description: Get outline (bounding) rectangle of Feature
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		protected override RectangleF GetOutLineRect()
		{
			return new RectangleF( PositionX, PositionY, m_SizeX, m_SizeY );
		}
		public CSIMRectangle(){}

		#region ISerializable Members
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMRectangle 
		/// Description: Deserialization constructor.
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMRectangle( SerializationInfo theInfo, StreamingContext theStreamingContext )
			: base( theInfo, theStreamingContext )
		{
			//Get the values from info and assign them to the appropriate properties
			SizeX = ( float )theInfo.GetValue( "SizeX", typeof( float ) );
			SizeY = ( float )theInfo.GetValue( "SizeY", typeof( float ) );
            SizeZ = ( float )theInfo.GetValue( "SizeZ", typeof( float ) );
		}

		//Serialization function.
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetObjectData 
		/// Description: Serialization function
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public override void GetObjectData( SerializationInfo theInfo, StreamingContext theStreamingContext )
		{
			base.GetObjectData( theInfo, theStreamingContext );
			theInfo.AddValue( "SizeX", SizeX );
			theInfo.AddValue( "SizeY", SizeY );
            theInfo.AddValue( "SizeZ", SizeZ );
		}
		#endregion
	}

	[Serializable]
	public class CSIMSegment : ISerializable
	{
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMRectangle 
		/// Description: Constructor
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMSegment(
			bool theLine,
			bool thePenDown,
			bool theClockwise,
			float thePositionX,
			float thePositionY,
			float theArcCentreX,
			float theArcCentreY
			)
		{
			m_ArcCentreX = theArcCentreX;
			m_ArcCentreY = theArcCentreY;
			m_Clockwise = theClockwise;
			m_Line = theLine;
			m_PenDown = thePenDown;
			m_PositionX = thePositionX;
			m_PositionY = thePositionY;
		}
		public CSIMSegment(){}

		private float m_ArcCentreX;
		private float m_ArcCentreY;
		private bool m_Clockwise;
		private bool m_Line;
		private bool m_PenDown;
		private float m_PositionX;
		private float m_PositionY;
		public float ArcCentreX { get { return m_ArcCentreX; } set { m_ArcCentreX = value; } }
		public float ArcCentreY { get { return m_ArcCentreY; } set { m_ArcCentreY = value; } }
		public bool Clockwise { get { return m_Clockwise; } set { m_Clockwise = value; } }
		public bool Line { get { return m_Line; } set { m_Line = value; } }
		public bool PenDown { get { return m_PenDown; } set { m_PenDown = value; } }
		public float PositionX { get { return m_PositionX; } set { m_PositionX = value; } }
		public float PositionY { get { return m_PositionY; } set { m_PositionY = value; } }

		#region ISerializable Members
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMSegment 
		/// Description: Deserialization constructor.
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMSegment( SerializationInfo theInfo, StreamingContext theStreamingContext )
		{
			//Get the values from info and assign them to the appropriate properties
			ArcCentreX = ( float )theInfo.GetValue( "ArcCentreX", typeof( float ) );
			ArcCentreY = ( float )theInfo.GetValue( "ArcCentreY", typeof( float ) );
			Clockwise = ( bool )theInfo.GetValue( "Clockwise", typeof( bool ) );
			Line = ( bool )theInfo.GetValue( "Line", typeof( bool ) );
			PenDown = ( bool )theInfo.GetValue( "PenDown", typeof( bool ) );
			PositionX = ( float )theInfo.GetValue( "PositionX", typeof( float ) );
			PositionY = ( float )theInfo.GetValue( "PositionY", typeof( float ) );
		}

		//Serialization function.
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetObjectData 
		/// Description: Serialization function
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public virtual void GetObjectData( SerializationInfo theInfo, StreamingContext theStreamingContext )
		{
			theInfo.AddValue( "ArcCentreX", ArcCentreX );
			theInfo.AddValue( "ArcCentreY", ArcCentreY );
			theInfo.AddValue( "Clockwise", Clockwise );
			theInfo.AddValue( "Line", Line );
			theInfo.AddValue( "PenDown", PenDown );
			theInfo.AddValue( "PositionX", PositionX );
			theInfo.AddValue( "PositionY", PositionY );
		}
		#endregion

	}
	[Serializable]
	public class CSIMShape : CSIMFeature
	{
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMShape 
		/// Description: Constructor
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMShape( int theReferenceID, float thePositionX, float thePositionY, float theRotation, short theNumberSegments )
			: base( theReferenceID, thePositionX, thePositionY, theRotation )
		{
			m_Type = t_FeatureType.e_CyberShape;
			m_Segments = new ArrayList( theNumberSegments );
		}
		private ArrayList m_Segments;
		public ArrayList Segments { get { return m_Segments; } set { m_Segments = value; } }
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetOutlineRect
		/// Description: Get outline (bounding) rectangle of Feature
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		protected override RectangleF GetOutLineRect()
		{
			float Top = ( m_Segments[ 0 ] as CSIMSegment ).PositionY;
			float Bottom = ( m_Segments[ 0 ] as CSIMSegment ).PositionY;
			float Left = ( m_Segments[ 0 ] as CSIMSegment ).PositionX;
			float Right = ( m_Segments[ 0 ] as CSIMSegment ).PositionX;
			foreach ( CSIMSegment ThisSegment in m_Segments )
			{
				Top = Math.Min( Top, ThisSegment.PositionY );
				Bottom = Math.Max( Top, ThisSegment.PositionY );
				Left = Math.Min( Top, ThisSegment.PositionX );
				Right = Math.Max( Top, ThisSegment.PositionX );
			}
			return new RectangleF( Left, Top, Right - Left, Bottom - Top );
		}
		public CSIMShape(){}
		#region ISerializable Members
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMShape 
		/// Description: Deserialization constructor.
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMShape( SerializationInfo theInfo, StreamingContext theStreamingContext )
			: base( theInfo, theStreamingContext )
		{
			//Get the values from info and assign them to the appropriate properties
			Segments = ( ArrayList )theInfo.GetValue( "Segments", typeof( ArrayList ) );
		}

		//Serialization function.
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: GetObjectData 
		/// Description: Serialization function
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public override void GetObjectData( SerializationInfo theInfo, StreamingContext theStreamingContext )
		{
			base.GetObjectData( theInfo, theStreamingContext );
			theInfo.AddValue( "Segments", Segments );
		}
		#endregion
	}
}
