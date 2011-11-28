using System;
using System.Collections.Generic;
using System.IO;
using System.Windows.Forms;
using System.Xml.Serialization;
using Cyber.MPanel;

namespace MPanelIO
{
    public class XmlToPanel
    {
        private static List<string> _featureErrorList = new List<string>();

        public static List<string> GetFeatureErrors()
        {
            return _featureErrorList;
        }

        /// <summary>
        /// This method will convert a CPanel to a CSIMPanel and serialize it
        /// to a XML file.
        /// </summary>
        /// <param name="xmlFilename">Path to write serialized CSIMPanel object.</param>
        /// <param name="panel">CPanel object to write to disk.</param>
        /// <returns>A boolean indicating success of conversion and writing to disk.</returns>
        static public bool CPanelToCSIMPanelXML(string xmlFilename, CPanel panel)
        {
            CSIMPanel simPanel = CPanelToCSIMPanel(panel);

            return SerializeCSIMPanelXML(xmlFilename, simPanel);
        }

        /// <summary>
        /// This function will read in a DEK::SIMService::CSIMPanel XML file and
        /// return a Cyber::SPIAPI::CPanel object.
        /// </summary>
        /// <param name="xmlFilename">Path to XML file containing serialized CSIMPanel object.</param>
        /// <param name="pixelSizeX">The CPanel object to return.</param>
        /// <param name="pixelSizeY">The CPanel object to return.</param>
        /// <returns>A boolean indicating if reading and converting to CPanel was succesful</returns>
        static public CPanel CSIMPanelXmlToCPanel(string xmlFilename, double pixelSizeX, double pixelSizeY)
        {
            CSIMPanel simPanel = DeserializeCSIMPanelXML(xmlFilename);

            return CSIMPanelToCPanel(simPanel, pixelSizeX, pixelSizeY);
        }

        /// <summary>
        /// This function is used to read a XML file that contains a 
        /// serialized CSIMPanel object.
        /// </summary>
        /// <param name="xmlFilename">Path to XML file containing serialized CSIMPanel object.</param>
        /// <returns>CSIMPanel object described in the file.</returns>
        static public CSIMPanel DeserializeCSIMPanelXML(string xmlFilename)
        {
            CSIMPanel simPanel = null;
            try
            {
                // Make sure even the construsctor runs inside a try-catch block
                System.Type[] PanelContainedClassTypes = new Type[10];
                PanelContainedClassTypes.SetValue(typeof(CSIMFeature), 0);
                PanelContainedClassTypes.SetValue(typeof(CSIMShape), 1);
                PanelContainedClassTypes.SetValue(typeof(CSIMDisc), 2);
                PanelContainedClassTypes.SetValue(typeof(CSIMRectangle), 3);
                PanelContainedClassTypes.SetValue(typeof(CSIMCross), 4);
                PanelContainedClassTypes.SetValue(typeof(CSIMDiamond), 5);
                PanelContainedClassTypes.SetValue(typeof(CSIMDonut), 6);
                PanelContainedClassTypes.SetValue(typeof(CSIMTriangle), 7);
                PanelContainedClassTypes.SetValue(typeof(CSIMCheckerPattern), 8);
                PanelContainedClassTypes.SetValue(typeof(CSIMSegment), 9);
                XmlSerializer PanelSerializer = new XmlSerializer(typeof(CSIMPanel), PanelContainedClassTypes);

                TextReader PanelFileReader = new StreamReader(xmlFilename);
                simPanel = (CSIMPanel)PanelSerializer.Deserialize(PanelFileReader);
                PanelFileReader.Close();

            }
            catch (Exception except)
            {
                MessageBox.Show("There was an exception reading the xml file: " + except.Message);
                return null;
            }

            if(simPanel == null)
            {
                MessageBox.Show("The panel should not be null!!!");
                return null;                
            }

            return simPanel;
        }

        /// <summary>
        /// This function is used to serialize a CSIMPanel object to 
        /// a XML file.
        /// </summary>
        /// <param name="xmlFilename">Path to XML file to write serialized CSIMPanel object.</param>
        /// <param name="panel">The CSIMPanel object to serialize.</param>
        /// <returns>A boolean indicating the serialization and file write success.</returns>
        static public bool SerializeCSIMPanelXML(string xmlFilename, CSIMPanel panel)
        {
            try
            {
                // Make sure even the constructor runs inside a try-catch block
                System.Type[] PanelContainedClassTypes = new Type[10];
                PanelContainedClassTypes.SetValue(typeof(CSIMFeature), 0);
                PanelContainedClassTypes.SetValue(typeof(CSIMShape), 1);
                PanelContainedClassTypes.SetValue(typeof(CSIMDisc), 2);
                PanelContainedClassTypes.SetValue(typeof(CSIMRectangle), 3);
                PanelContainedClassTypes.SetValue(typeof(CSIMCross), 4);
                PanelContainedClassTypes.SetValue(typeof(CSIMDiamond), 5);
                PanelContainedClassTypes.SetValue(typeof(CSIMDonut), 6);
                PanelContainedClassTypes.SetValue(typeof(CSIMTriangle), 7);
                PanelContainedClassTypes.SetValue(typeof(CSIMCheckerPattern), 8);
                PanelContainedClassTypes.SetValue(typeof(CSIMSegment), 9);
                XmlSerializer PanelSerializer = new XmlSerializer(typeof(CSIMPanel), PanelContainedClassTypes);

                TextWriter PanelFileWriter = new StreamWriter(xmlFilename);
                PanelSerializer.Serialize(PanelFileWriter, panel);
                PanelFileWriter.Close();
            }
            catch (Exception except)
            {
                //throw;
                MessageBox.Show("There was an exception writing the xml file: " + except.Message);
                return false;
            }

            return true;
        }

        /// <summary>
        /// This method will convert a Cyber::SPIAPI::CPanel object to a 
        /// DEK::SIMService::CSIMPanel object.
        /// </summary>
        /// <param name="panel">CPanel object to convert.</param>
        /// <returns>The converted CSIMPanel object, or NULL if it failed.</returns>
        static private CSIMPanel CPanelToCSIMPanel(CPanel panel)
        {
            try
            {
                CSIMPanel simPanel = new CSIMPanel(panel.Name, 
                     ToCSIMPanelUnits(panel.PanelSizeX), ToCSIMPanelUnits(panel.PanelSizeY));

                CFeature fid = panel.GetFirstFiducial();
                while(fid != null)
                {
                    switch (fid.Type)
                    {
                        case CFeature.ShapeType.CyberShape:
                            {
                                CCyberShape cs = (CCyberShape) fid;
                                CSIMShape simShape = new CSIMShape(cs.ReferenceID, 
                                    ToCSIMPanelUnits(cs.PositionX), ToCSIMPanelUnits(cs.PositionY),
                                    (float) cs.Rotation, (short) cs.GetNumberSegments());

                                CCyberSegment s = cs.GetFirstSegment();
                                while(s != null)
                                {
                                    CSIMSegment simSegment = new CSIMSegment(s.Line, s.PenDown, s.ClockwiseArc,
                                        ToCSIMPanelUnits(s.PositionX), ToCSIMPanelUnits(s.PositionY), 
                                        ToCSIMPanelUnits(s.ArcX), ToCSIMPanelUnits(s.ArcY));

                                    simShape.Segments.Add(simSegment);

                                    s = cs.GetNextSegment();
                                }
                                simPanel.Fiducials.Add(simShape);
                                break;
                            }
                        case CFeature.ShapeType.Disc:
                            {
                                CDisc d = (CDisc) fid;
                                CSIMDisc disc = new CSIMDisc(d.ReferenceID, 
                                    ToCSIMPanelUnits(d.PositionX), ToCSIMPanelUnits(d.PositionY), 
                                    (float)d.Rotation, ToCSIMPanelUnits(d.Diameter));
                                simPanel.Fiducials.Add(disc);
                                break;
                            }
                        case CFeature.ShapeType.Rectangle:
                            {
                                CRectangle r = (CRectangle) fid;
                                CSIMRectangle rectangle = new CSIMRectangle(r.ReferenceID, 
                                    ToCSIMPanelUnits(r.PositionX), ToCSIMPanelUnits(r.PositionY),
                                    (float) r.Rotation, ToCSIMPanelUnits(r.SizeX), ToCSIMPanelUnits(r.SizeY));
                                simPanel.Fiducials.Add(rectangle);
                                break;
                            }
                        case CFeature.ShapeType.Cross:
                            {
                                CCross c = (CCross)fid;
                                CSIMCross cross = new CSIMCross(c.ReferenceID,
                                    ToCSIMPanelUnits(c.PositionX), ToCSIMPanelUnits(c.PositionY),
                                    (float)c.Rotation, ToCSIMPanelUnits(c.SizeX), ToCSIMPanelUnits(c.SizeY), ToCSIMPanelUnits(c.LegSizeX), ToCSIMPanelUnits(c.LegSizeY));
                                simPanel.Fiducials.Add(cross);
                                break;
                            }

                        case CFeature.ShapeType.Diamond:
                            {
                                CDiamond di = (CDiamond)fid;
                                CSIMDiamond diamond = new CSIMDiamond(di.ReferenceID,
                                    ToCSIMPanelUnits(di.PositionX), ToCSIMPanelUnits(di.PositionY),
                                    (float)di.Rotation, ToCSIMPanelUnits(di.SizeX), ToCSIMPanelUnits(di.SizeY));
                                simPanel.Fiducials.Add(diamond);
                                break;
                            }

                        case CFeature.ShapeType.Donut:
                            {
                                CDonut t = (CDonut)fid;
                                CSIMDonut donut = new CSIMDonut(t.ReferenceID,
                                    ToCSIMPanelUnits(t.PositionX), ToCSIMPanelUnits(t.PositionY),
                                    (float)t.Rotation, ToCSIMPanelUnits(t.DiameterInside), ToCSIMPanelUnits(t.DiameterOutside));
                                simPanel.Fiducials.Add(donut);
                                break;
                            }

                        case CFeature.ShapeType.Triangle:
                            {
                                CTriangle t = (CTriangle) fid;
                                CSIMTriangle triangle = new CSIMTriangle(t.ReferenceID, 
                                    ToCSIMPanelUnits(t.PositionX), ToCSIMPanelUnits(t.PositionY),
                                    (float)t.Rotation, ToCSIMPanelUnits(t.SizeX), ToCSIMPanelUnits(t.SizeY), ToCSIMPanelUnits(t.OffsetX));
                                simPanel.Fiducials.Add(triangle);
                                break;
                            }


                        case CFeature.ShapeType.CheckerPattern:
                            {
                                CCheckerPattern t = (CCheckerPattern)fid;
                                CSIMCheckerPattern checkerP = new CSIMCheckerPattern(t.ReferenceID,
                                    ToCSIMPanelUnits(t.PositionX), ToCSIMPanelUnits(t.PositionY),
                                    (float)t.Rotation, ToCSIMPanelUnits(t.SizeX), ToCSIMPanelUnits(t.SizeY));
                                simPanel.Fiducials.Add(checkerP);
                                break;
                            }

                       case CFeature.ShapeType.Undefined:
                        default:
                            {
                                break;
                            }
                    }

                    fid = panel.GetNextFiducial();
                }

                CFeature feature = panel.GetFirstFeature();
                while (feature != null)
                {
                    switch (feature.Type)
                    {
                        case CFeature.ShapeType.CyberShape:
                            {
                                CCyberShape cs = (CCyberShape)feature;
                                CSIMShape simShape = new CSIMShape(cs.ReferenceID,
                                    ToCSIMPanelUnits(cs.PositionX), ToCSIMPanelUnits(cs.PositionY),
                                    (float)cs.Rotation, (short)cs.GetNumberSegments());

                                CCyberSegment s = cs.GetFirstSegment();
                                while (s != null)
                                {
                                    CSIMSegment simSegment = new CSIMSegment(s.Line, s.PenDown, s.ClockwiseArc,
                                        ToCSIMPanelUnits(s.PositionX), ToCSIMPanelUnits(s.PositionY),
                                        ToCSIMPanelUnits(s.ArcX), ToCSIMPanelUnits(s.ArcY));

                                    simShape.Segments.Add(simSegment);

                                    s = cs.GetNextSegment();
                                }
                                simPanel.Features.Add(simShape);
                                break;
                            }
                        case CFeature.ShapeType.Disc:
                            {
                                CDisc d = (CDisc)feature;
                                CSIMDisc disc = new CSIMDisc(d.ReferenceID,
                                    ToCSIMPanelUnits(d.PositionX), ToCSIMPanelUnits(d.PositionY),
                                    (float)d.Rotation, ToCSIMPanelUnits(d.Diameter));
                                simPanel.Features.Add(disc);
                                break;
                            }
                        case CFeature.ShapeType.Rectangle:
                            {
                                CRectangle r = (CRectangle)feature;
                                CSIMRectangle rectangle = new CSIMRectangle(r.ReferenceID,
                                    ToCSIMPanelUnits(r.PositionX), ToCSIMPanelUnits(r.PositionY),
                                    (float)r.Rotation, ToCSIMPanelUnits(r.SizeX), ToCSIMPanelUnits(r.SizeY));
                                simPanel.Features.Add(rectangle);
                                break;
                            }
                        case CFeature.ShapeType.Cross:
                        case CFeature.ShapeType.Diamond:
                        case CFeature.ShapeType.Donut:
                        case CFeature.ShapeType.Triangle:
                            {
                                CFeature f = (CFeature)feature;
                                CSIMFeature feat = new CSIMFeature(f.ReferenceID,
                                    ToCSIMPanelUnits(f.PositionX), ToCSIMPanelUnits(f.PositionY),
                                    (float)f.Rotation);
                                simPanel.Fiducials.Add(feat);
                                break;
                            }
                        case CFeature.ShapeType.Undefined:
                        default:
                            {
                                break;
                            }
                    }

                    feature = panel.GetNextFeature();
                }


                return simPanel;
            }
            catch (Exception exception)
            {
                //throw;
                MessageBox.Show("An exception occured converting CPanel to CSIMPanel: " + exception.Message);
                return null;
            }
        }

        /// <summary>
        /// This function will convert a DEK::SIMService::CSIMPanel object to 
        /// a Cyber::SPIAPI::CPanel object.
        /// </summary>
        /// <note>The CSIMPanel is in millimeters, where the CPanel should be
        /// in meters.  This function will perform the conversion.
        /// </note>
        /// <param name="simPanel"></param>
        /// <param name="pixelSizeX"></param>
        /// <param name="pixelSizeY"></param>
        /// <returns></returns>
        static private CPanel CSIMPanelToCPanel(CSIMPanel simPanel, double pixelSizeX, double pixelSizeY)
        {
            if (simPanel == null)
                return null;

            CPanel panel;
            try
            {
                _featureErrorList.Clear();

                panel = new CPanel(
                    ToCPanelUnits(simPanel.PanelSize.ToPointF().X),
                    ToCPanelUnits(simPanel.PanelSize.ToPointF().Y),
                    pixelSizeX, pixelSizeY);
 
                panel.Name = simPanel.PanelName;

                if(panel.PanelSizeX < 0.000001 || panel.PanelSizeY < 0.000001)
                    throw new ArgumentException("Invalid panel size");

                /// Add CSIMPanel's fiducials to CPanel's fiducials
                foreach (CSIMFeature simFid in simPanel.Fiducials)
                {
                    try
                    {
                        switch (simFid.Type)
                        {
                            case t_FeatureType.e_CyberShape:
                                {
                                    CSIMShape shape = (CSIMShape)simFid;
                                    CCyberShape cyberShape = new CCyberShape(shape.ReferenceID,
                                        ToCPanelUnits(shape.PositionX), ToCPanelUnits(shape.PositionY),
                                        shape.Rotation);

                                    foreach (CSIMSegment s in shape.Segments)
                                    {
                                        CCyberSegment segment = new CCyberSegment(s.Line, s.PenDown, s.Clockwise,
                                            ToCPanelUnits(s.PositionX), ToCPanelUnits(s.PositionY),
                                            ToCPanelUnits(s.ArcCentreX), ToCPanelUnits(s.ArcCentreY));
                                        cyberShape.AddSegment(segment);
                                    }

                                    panel.AddFiducial(cyberShape);
                                    break;
                                }
                            case t_FeatureType.e_Disc:
                                {
                                    CSIMDisc simDisc = (CSIMDisc)simFid;
                                    CDisc disc = new CDisc(simDisc.ReferenceID,
                                        ToCPanelUnits(simDisc.PositionX), ToCPanelUnits(simDisc.PositionY),
                                        ToCPanelUnits(simDisc.Diameter));
                                    panel.AddFiducial(disc);
                                    break;
                                }
                            case t_FeatureType.e_Rectangle:
                                {
                                    CSIMRectangle simRect = (CSIMRectangle)simFid;
                                    CRectangle rect = new CRectangle(simRect.ReferenceID,
                                        ToCPanelUnits(simRect.PositionX), ToCPanelUnits(simRect.PositionY), simRect.Rotation,
                                        ToCPanelUnits(simRect.SizeX), ToCPanelUnits(simRect.SizeY));
                                    panel.AddFiducial(rect);
                                    break;
                                }


                            case t_FeatureType.e_Cross:
                                {
                                    CSIMCross simCro = (CSIMCross)simFid;
                                    CCross cro = new CCross(simCro.ReferenceID,
                                        ToCPanelUnits(simCro.PositionX), ToCPanelUnits(simCro.PositionY), simCro.Rotation,
                                        ToCPanelUnits(simCro.SizeX), ToCPanelUnits(simCro.SizeY), ToCPanelUnits(simCro.LegSizeX), ToCPanelUnits(simCro.LegSizeY));
                                    panel.AddFiducial(cro);
                                    break;
                                }

                            case t_FeatureType.e_Diamond:
                                {
                                    CSIMDiamond simDia = (CSIMDiamond)simFid;
                                    CRectangle dia = new CRectangle(simDia.ReferenceID,
                                        ToCPanelUnits(simDia.PositionX), ToCPanelUnits(simDia.PositionY), simDia.Rotation,
                                        ToCPanelUnits(simDia.SizeX), ToCPanelUnits(simDia.SizeY));
                                    panel.AddFiducial(dia);
                                    break;
                                }
                            case t_FeatureType.e_Donut:
                                {
                                    CSIMDonut simDon = (CSIMDonut)simFid;
                                    CDonut don = new CDonut(simDon.ReferenceID,
                                        ToCPanelUnits(simDon.PositionX), ToCPanelUnits(simDon.PositionY),
                                        ToCPanelUnits(simDon.DiameterInside), ToCPanelUnits(simDon.DiameterOutside));
                                    panel.AddFiducial(don);
                                    break;
                                }
                            case t_FeatureType.e_Triangle:
                                {
                                    CSIMTriangle simTri = (CSIMTriangle)simFid;
                                    CTriangle tri = new CTriangle(simTri.ReferenceID,
                                        ToCPanelUnits(simTri.PositionX), ToCPanelUnits(simTri.PositionY), simTri.Rotation,
                                        ToCPanelUnits(simTri.SizeX), ToCPanelUnits(simTri.SizeY),ToCPanelUnits(simTri.Offset));
                                    panel.AddFiducial(tri);
                                    break;
                                }

                            case t_FeatureType.e_CheckerPattern:
                                {
                                    CSIMCheckerPattern simCheckerP = (CSIMCheckerPattern)simFid;
                                    CCheckerPattern checkerP = new CCheckerPattern(simCheckerP.ReferenceID,
                                        ToCPanelUnits(simCheckerP.PositionX), ToCPanelUnits(simCheckerP.PositionY), simCheckerP.Rotation,
                                        ToCPanelUnits(simCheckerP.SizeX), ToCPanelUnits(simCheckerP.SizeY));
                                    panel.AddFiducial(checkerP);
                                    break;
                                }
                            default:
                                {
                                    break;
                                }
                        }
                    }
                    catch (Exception e)
                    {
                        throw new ArgumentException("Failed to convert fiducial feature " + simFid.ReferenceID, e);
                    }
                }

                
                /// Add CSIMPanel's CSIMFeatures to CPanel as CFeatures
                foreach (CSIMFeature simFeature in simPanel.Features)
                {
                    try
                    {
                        switch (simFeature.Type)
                        {
                            case t_FeatureType.e_CyberShape:
                                {
                                    CSIMShape shape = (CSIMShape) simFeature;
                                    CCyberShape cyberShape = new CCyberShape(shape.ReferenceID,
                                                                             ToCPanelUnits(shape.PositionX),
                                                                             ToCPanelUnits(shape.PositionY),
                                                                             shape.Rotation);

                                    foreach (CSIMSegment s in shape.Segments)
                                    {
                                        CCyberSegment segment = new CCyberSegment(s.Line, s.PenDown, s.Clockwise,
                                                                                  ToCPanelUnits(s.PositionX),
                                                                                  ToCPanelUnits(s.PositionY),
                                                                                  ToCPanelUnits(s.ArcCentreX),
                                                                                  ToCPanelUnits(s.ArcCentreY));
                                        cyberShape.AddSegment(segment);
                                    }

                                    panel.AddFeature(cyberShape);
                                    break;
                                }
                            case t_FeatureType.e_Disc:
                                {
                                    CSIMDisc simDisc = (CSIMDisc) simFeature;
                                    CDisc disc = new CDisc(simDisc.ReferenceID,
                                                           ToCPanelUnits(simDisc.PositionX),
                                                           ToCPanelUnits(simDisc.PositionY),
                                                           ToCPanelUnits(simDisc.Diameter));
                                    panel.AddFeature(disc);
                                    break;
                                }
                            case t_FeatureType.e_Rectangle:
                                {
                                    CSIMRectangle simRect = (CSIMRectangle) simFeature;
                                    CRectangle rect = new CRectangle(simRect.ReferenceID,
                                                                     ToCPanelUnits(simRect.PositionX),
                                                                     ToCPanelUnits(simRect.PositionY), simRect.Rotation,
                                                                     ToCPanelUnits(simRect.SizeX),
                                                                     ToCPanelUnits(simRect.SizeY),
                                                                     ToCPanelUnits(simRect.SizeZ));
                                    panel.AddFeature(rect);
                                    break;
                                }
                            case t_FeatureType.e_Cross:
                            case t_FeatureType.e_Diamond:
                            case t_FeatureType.e_Donut:
                            case t_FeatureType.e_Triangle:
                            default:
                                {
                                    break;
                                }
                        }
                    }
                    catch (Exception e)
                    {
                        throw new ArgumentException("Failed to convert fiducial feature " + simFeature.ReferenceID, e);
                    }
                }
            }
            catch (Exception exception)
            {
                throw new Exception("An exception was thrown converting the Panel: " + simPanel.PanelName, exception);
            }


            return panel;
        }

        private const float UnitsScalingFactor = 1000.0f; // The input Panel is always in mm, the one sent to SIM is always meters.
        /// <summary>
        /// Convert value in Units used by DEK(mm) to value in Units used by CSPIAPI (meters).
        /// </summary>
        private static float ToCPanelUnits(float theValueInMM)
        {
            return theValueInMM / UnitsScalingFactor;
        }

        /// <summary>
        /// Convert value in units used by CSPIAPI (meters) to value in units used by DEK (mm).
        /// </summary>
        /// <param name="theValueInMeters"></param>
        /// <returns></returns>
        private static float ToCSIMPanelUnits(double theValueInMeters)
        {
            return ((float)theValueInMeters)*UnitsScalingFactor;
        }



    }
}
