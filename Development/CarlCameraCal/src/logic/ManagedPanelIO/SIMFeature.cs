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
        e_DiamondFrame = 2,
		e_Disc = 3,
		e_Donut = 4,
		e_Rectangle = 5,
        e_RectangleFrame =6,
		e_Triangle = 7,
        e_EquilateralTriangleFrame = 8,
        e_CheckerPattern = 9,
		e_CyberShape = 10

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
    [System.Xml.Serialization.XmlInclude(typeof(CSIMDisc))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMRectangle))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMRectangleFrame))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMCross))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMDiamond))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMDiamondFrame))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMDonut))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMTriangle))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMEquilateralTriangleFrame))]
    [System.Xml.Serialization.XmlInclude(typeof(CSIMCheckerPattern))]

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
    public class CSIMRectangleFrame : CSIMFeature
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMRectangleFrame 
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMRectangleFrame(int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSizeX, float theSizeY, float theThickness)
            : base(theReferenceID, thePositionX, thePositionY, theRotation)
        {
            m_Type = t_FeatureType.e_RectangleFrame;
            m_SizeX = theSizeX;
            m_SizeY = theSizeY;
            m_Thickness = theThickness;
        }
        private float m_SizeX;
        private float m_SizeY;
        private float m_Thickness;
        public float SizeX { get { return m_SizeX; } set { m_SizeX = value; } }
        public float SizeY { get { return m_SizeY; } set { m_SizeY = value; } }
        public float Thickness { get { return m_Thickness; } set { m_Thickness = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, m_SizeX, m_SizeY);
        }
        public CSIMRectangleFrame() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMRectangleFrame 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMRectangleFrame(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            SizeX = (float)theInfo.GetValue("SizeX", typeof(float));
            SizeY = (float)theInfo.GetValue("SizeY", typeof(float));
            Thickness = (float)theInfo.GetValue("Thickness", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("SizeX", SizeX);
            theInfo.AddValue("SizeY", SizeY);
            theInfo.AddValue("Thickness", Thickness);
        }
        #endregion
    }

    [Serializable]
    public class CSIMCross : CSIMFeature
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMCross 
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMCross(int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSizeX, float theSizeY, float theLegSizeX,float theLegSizeY )
            : base(theReferenceID, thePositionX, thePositionY, theRotation)
        {
            m_Type = t_FeatureType.e_Cross;
            m_SizeX = theSizeX;
            m_SizeY = theSizeY;
            m_LegSizeX = theLegSizeX;
            m_LegSizeY = theLegSizeY;
        }
        private float m_SizeX;
        private float m_SizeY;
        private float m_LegSizeX;
        private float m_LegSizeY;
        public float SizeX { get { return m_SizeX; } set { m_SizeX = value; } }
        public float SizeY { get { return m_SizeY; } set { m_SizeY = value; } }
        public float LegSizeX { get { return m_LegSizeX; } set {m_LegSizeX = value; } }
        public float LegSizeY  { get { return m_LegSizeY; } set {m_LegSizeY = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, m_SizeX, m_SizeY);
        }
        public CSIMCross() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMCross 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMCross(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            SizeX = (float)theInfo.GetValue("SizeX", typeof(float));
            SizeY = (float)theInfo.GetValue("SizeY", typeof(float));
            LegSizeX = (float)theInfo.GetValue("LegSizeX", typeof(float));
            LegSizeY = (float)theInfo.GetValue("LegSizeY", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("SizeX", SizeX);
            theInfo.AddValue("SizeY", SizeY);
            theInfo.AddValue("LegSizeX", LegSizeX);
            theInfo.AddValue("LegSizeY", LegSizeY);
        }
        #endregion
    }

    [Serializable]
    public class CSIMDiamond : CSIMFeature
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMDiamond 
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMDiamond(int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSizeX, float theSizeY)
            : base(theReferenceID, thePositionX, thePositionY, theRotation)
        {
            m_Type = t_FeatureType.e_Diamond;
            m_SizeX = theSizeX;
            m_SizeY = theSizeY;
        }
        private float m_SizeX;
        private float m_SizeY;
        public float SizeX { get { return m_SizeX; } set { m_SizeX = value; } }
        public float SizeY { get { return m_SizeY; } set { m_SizeY = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, m_SizeX, m_SizeY);
        }
        public CSIMDiamond() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMDiamond 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMDiamond(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            SizeX = (float)theInfo.GetValue("SizeX", typeof(float));
            SizeY = (float)theInfo.GetValue("SizeY", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("SizeX", SizeX);
            theInfo.AddValue("SizeY", SizeY);
        }
        #endregion
    }

    [Serializable]
    public class CSIMDiamondFrame : CSIMFeature
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMDiamondFrame 
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMDiamondFrame(int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSizeX, float theSizeY,float theThickness)
            : base(theReferenceID, thePositionX, thePositionY, theRotation)
        {
            m_Type = t_FeatureType.e_DiamondFrame;
            m_SizeX = theSizeX;
            m_SizeY = theSizeY;
            m_Thickness = theThickness;
        }
        private float m_SizeX;
        private float m_SizeY;
        private float m_Thickness;
        public float SizeX { get { return m_SizeX; } set { m_SizeX = value; } }
        public float SizeY { get { return m_SizeY; } set { m_SizeY = value; } }
        public float Thickness { get { return m_Thickness; } set {m_Thickness = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, m_SizeX, m_SizeY);
        }
        public CSIMDiamondFrame() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMDiamond 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMDiamondFrame(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            SizeX = (float)theInfo.GetValue("SizeX", typeof(float));
            SizeY = (float)theInfo.GetValue("SizeY", typeof(float));
            Thickness = (float)theInfo.GetValue("Thickness", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("SizeX", SizeX);
            theInfo.AddValue("SizeY", SizeY);
            theInfo.AddValue("Thickness", Thickness);
        }
        #endregion
    }

    [Serializable]
    public class CSIMDonut : CSIMFeature
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMDonut
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMDonut(int theReferenceID, float thePositionX, float thePositionY, float theRotation,  float theDiameterInside, float theDiameterOutside)
            : base(theReferenceID, thePositionX, thePositionY, theRotation)
        {
            m_Type = t_FeatureType.e_Donut;
            m_DiameterInside = theDiameterInside;
            m_DiameterOutside = theDiameterOutside;
        }
        private float m_DiameterInside;
        private float m_DiameterOutside;
        public float DiameterInside { get { return m_DiameterInside; } set { m_DiameterInside = value; } }
        public float DiameterOutside { get { return m_DiameterOutside; } set { m_DiameterOutside = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, m_DiameterOutside, m_DiameterOutside);
        }
        public CSIMDonut() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMDonut 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMDonut(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            DiameterInside = (float)theInfo.GetValue("DiameterInside", typeof(float));
            DiameterOutside = (float)theInfo.GetValue("DiameterOutside", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("DiameterInside", DiameterInside);
            theInfo.AddValue("DiameterOutside", DiameterOutside);
        }
        #endregion
    }

    [Serializable]
    public class CSIMTriangle : CSIMFeature
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMTriangle
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMTriangle(int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSizeX, float theSizeY, float theOffset)
            : base(theReferenceID, thePositionX, thePositionY, theRotation)
        {
            m_Type = t_FeatureType.e_Triangle;
            m_SizeX = theSizeX;
            m_SizeY = theSizeY;
            m_Offset = theOffset;
        }
        private float m_SizeX;
        private float m_SizeY;
        private float m_Offset;
        public float SizeX { get { return m_SizeX; } set { m_SizeX = value; } }
        public float SizeY { get { return m_SizeY; } set { m_SizeY = value; } }
        public float Offset { get { return m_Offset; } set { m_Offset = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, m_SizeX, m_SizeY);
        }
        public CSIMTriangle() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMTriangle 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMTriangle(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            SizeX = (float)theInfo.GetValue("SizeX", typeof(float));
            SizeY = (float)theInfo.GetValue("SizeY", typeof(float));
            Offset = (float)theInfo.GetValue("Offset", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("SizeX", SizeX);
            theInfo.AddValue("SizeY", SizeY);
            theInfo.AddValue("Offset", Offset);
        }
        #endregion
    }

    [Serializable]
    public class CSIMEquilateralTriangleFrame : CSIMTriangle
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMEquilateralTriangleFrame
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMEquilateralTriangleFrame(int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSize, float theThickness)
            : base(theReferenceID, thePositionX, thePositionY, theRotation, theSize, (float)(theSize*Math.Sqrt(3)/2.0), (float)(theSize / 2.0))
        {
            m_Type = t_FeatureType.e_EquilateralTriangleFrame;
            m_Size = theSize;
            m_Thickness = theThickness;
        }

        private float m_Thickness;
        private float m_Size;
        public float Size { get { return m_Size; } set { m_Size = value; } }
        public float Thickness { get { return m_Thickness; } set { m_Thickness = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, SizeX, SizeY);
        }
        public CSIMEquilateralTriangleFrame() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMEquilateralTriangleFrame 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMEquilateralTriangleFrame(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            Size = (float)theInfo.GetValue("Size", typeof(float));
            Thickness = (float)theInfo.GetValue("Thickness", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("Size", Size);
            theInfo.AddValue("Thickness", Thickness);
        }
        #endregion
    }

    [Serializable]
    public class CSIMCheckerPattern : CSIMFeature
    {
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMCheckerPatytern
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMCheckerPattern(int theReferenceID, float thePositionX, float thePositionY, float theRotation, float theSizeX, float theSizeY)
            : base(theReferenceID, thePositionX, thePositionY, theRotation)
        {
            m_Type = t_FeatureType.e_CheckerPattern;
            m_SizeX = theSizeX;
            m_SizeY = theSizeY;
        }
        private float m_SizeX;
        private float m_SizeY;
        public float SizeX { get { return m_SizeX; } set { m_SizeX = value; } }
        public float SizeY { get { return m_SizeY; } set { m_SizeY = value; } }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetOutlineRect
        /// Description: Get outline (bounding) rectangle of Feature
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        protected override RectangleF GetOutLineRect()
        {
            return new RectangleF(PositionX, PositionY, m_SizeX, m_SizeY);
        }
        public CSIMCheckerPattern() { }

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMCheckerPattern 
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMCheckerPattern(SerializationInfo theInfo, StreamingContext theStreamingContext)
            : base(theInfo, theStreamingContext)
        {
            //Get the values from info and assign them to the appropriate properties
            SizeX = (float)theInfo.GetValue("SizeX", typeof(float));
            SizeY = (float)theInfo.GetValue("SizeY", typeof(float));
        }

        //Serialization function.
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public override void GetObjectData(SerializationInfo theInfo, StreamingContext theStreamingContext)
        {
            base.GetObjectData(theInfo, theStreamingContext);
            theInfo.AddValue("SizeX", SizeX);
            theInfo.AddValue("SizeY", SizeY);
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
