using System;
using System.IO;
using Cyber.SPIAPI;

namespace CPanelIO
{
    public class CsvToPanel
    {
        public static bool parseCSV(string path, ref CPanel panel)
        {
            panel.ClearFeatures();
            panel.ClearFiducials();

            try
            {
                using (StreamReader readFile = new StreamReader(path))
                {

                    string line;

                    while ((line = readFile.ReadLine()) != null)
                    {
                        bool result;
                        double length;
                        string[] row;
                        int referenceID = -1;
                        double positionX = -1.0;
                        double positionY = -1.0;
                        double rotation = -1.0;
                        double diameter = -1.0;

                        row = line.Split(',');
                        switch (row[0])
                        {
                            case "LengthX":
                                result = Double.TryParse(row[1], out length);
                                if (result) panel.PanelSizeX = (float) length;
                                else panel.PanelSizeX = (float)0.0;
                                break;
                            case "LengthY":
                                result = Double.TryParse(row[1], out length);
                                if (result) panel.PanelSizeY = (float)length;
                                else panel.PanelSizeY = (float)0.0;
                                break;
                            case "Feature":

                                result = Int32.TryParse(row[1], out referenceID);
                                if (!result) break;
                                result = Double.TryParse(row[3], out positionX);
                                if (!result) break;
                                result = Double.TryParse(row[4], out positionY);
                                if (!result) break;
                                result = Double.TryParse(row[5], out rotation);
                                if (!result) break;

                                switch (row[6])
                                {
                                    case "Rectangle":
                                        double sizeX, sizeY;
                                        result = Double.TryParse(row[7], out sizeX);
                                        if (!result) break;
                                        result = Double.TryParse(row[8], out sizeY);
                                        if (!result) break;
                                        CRectangle theRectangle = new CRectangle(referenceID,
                                                                                 positionX, positionY, rotation, sizeX,
                                                                                 sizeY);
                                        panel.AddFeature(theRectangle);
                                        break;
                                    case "Disc":
                                        result = Double.TryParse(row[7], out diameter);

                                        // we actually just read in the radius
                                        diameter *= 2;

                                        if (!result) break;
                                        CDisc theDisc = new CDisc(referenceID, positionX, positionY, diameter);
                                        panel.AddFeature(theDisc);
                                        break;
                                    default:
                                        break;
                                }
                                break;
                            case "Fiducial":
                                result = Int32.TryParse(row[1], out referenceID);
                                if (!result) break;
                                result = Double.TryParse(row[3], out positionX);
                                if (!result) break;
                                result = Double.TryParse(row[4], out positionY);
                                if (!result) break;
                                result = Double.TryParse(row[5], out rotation);
                                if (!result) break;
                                Double.TryParse(row[7], out diameter);

                                // we actually just read in the radius
                                diameter *= 2;

                                CDisc theFid = new CDisc(referenceID,
                                                         positionX,
                                                         positionY,
                                                         diameter);

                                panel.AddFiducial(theFid);

                                break;
                            default:
                                break;
                        }

                        //parsedData.Add(row);
                    }
                }
            }
            catch (Exception e)
            {
                System.Windows.Forms.MessageBox.Show(e.Message);
            }

#warning - you aren't returning the validatation?
            return true;
        }


    }
}
