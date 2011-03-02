////////////////////////////////////////////////////////////////////////////////
// 
// Filename: SIMPanel.cs
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
// Code in this file conforms to the standards described in the C/C++
// Coding Standards Document, Specification DSS111.
////////////////////////////////////////////////////////////////////////////////
using System;
using System.Collections;
using System.Drawing;
using System.Runtime.Serialization;

namespace CPanelIO
{

    //////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// Name: CSIMPanel class
    /// Description: Base class for Panel that the SIM can inspect. Serializable so 
    /// can be sent to remote SIM. Units are assumed to be in Millimetres
    /// </summary>
    //////////////////////////////////////////////////////////////////////////////
    [Serializable]
    public class CSIMPanel : System.Runtime.Serialization.ISerializable
    {
        #region Attribute and corresponding properties
		private string m_PanelName;
		private int m_PanelOrigin; // enum of front left, back right etc tbd
        private SizeF m_PanelSize;
        private float m_Rotation;
        private ArrayList m_Features;
        private ArrayList m_Fiducials;
		public string PanelName { get { return m_PanelName; } set { m_PanelName = value; } }
		public int PanelOrigin { get { return m_PanelOrigin; } set { m_PanelOrigin = value; } }
		public SizeF PanelSize { get { return m_PanelSize; } set { m_PanelSize = value; } }
        public float Rotation { get { return m_Rotation; } set { m_Rotation = value; } }
        public ArrayList Features { get { return m_Features; } set { m_Features = value; } }
        public ArrayList Fiducials { get { return m_Fiducials; } set { m_Fiducials = value; } }
        #endregion Attribute and corresponding properties

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMPanel 
        /// Description: Constructor
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMPanel( string thePanelName, float theLengthX, float theLengthY )
        {
			m_PanelName = thePanelName;
            m_Features = new ArrayList();
            m_Fiducials = new ArrayList();
            m_PanelSize.Width = theLengthX;
            m_PanelSize.Height = theLengthY;
        }
		//////////////////////////////////////////////////////////////////////////////
		/// <summary>
		/// Name: CSIMPanel 
		/// Description: Default Constructor - needed for serialization
		/// </summary>
		//////////////////////////////////////////////////////////////////////////////
		public CSIMPanel()
		{
		}

        #region ISerializable Members
        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: CSIMPanel
        /// Description: Deserialization constructor.
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public CSIMPanel( SerializationInfo theInfo, StreamingContext theStreamingContext )
        {
            //Get the values from info and assign them to the appropriate properties
			PanelName = theInfo.GetString( "PanelName" );
			PanelOrigin = theInfo.GetInt32( "PanelOrigin" );
			PanelSize = ( SizeF )theInfo.GetValue( "PanelSize", typeof( SizeF ) );
            Rotation = ( float )theInfo.GetValue( "Rotation", typeof( float ) );
            Features = ( ArrayList )theInfo.GetValue( "Features", typeof( ArrayList ) );
            Fiducials = ( ArrayList )theInfo.GetValue( "Fiducials", typeof( ArrayList ) );
        }

        //////////////////////////////////////////////////////////////////////////////
        /// <summary>
        /// Name: GetObjectData 
        /// Description: Serialization function
        /// </summary>
        //////////////////////////////////////////////////////////////////////////////
        public void GetObjectData( SerializationInfo theInfo, StreamingContext theStreamingContext )
        {
			theInfo.AddValue( "PanelName", PanelName );
			theInfo.AddValue( "PanelOrigin", PanelOrigin );
            theInfo.AddValue( "PanelSize", PanelSize );
            theInfo.AddValue( "Rotation", Rotation );
            theInfo.AddValue( "Features", Features );
            theInfo.AddValue( "Fiducials", Fiducials );
        }
    }
        #endregion

}
