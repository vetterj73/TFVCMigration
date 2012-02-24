<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:msxsl="urn:schemas-microsoft-com:xslt" xmlns:SgisScript="http://www.soilteq.com/SGIS/Script">
	<xsl:output method="html" version="4.0" indent="yes"/>
	<xsl:template match="manifest">
		<html>
			<head>
				<title>EPV System Software Dependency Report</title>
				<style TYPE="text/css">
          body {
          font-size: x-small;
          font-family:Arial, Helvetica, sans-serif;
          margin:0%;
          }

          h2 {
          margin-top: 1%;
          margin-bottom: 0%;
          margin-left: 2%;
          }

          h3 {
          margin-top: 1%;
          margin-bottom: 0%;
          margin-left: 4%;
          }

          h4 {
          margin-top: 0%;
          margin-bottom: 0%;
          margin-left: 6%;
          }
          td
          {
          font-size: x-small;
          font-family:Arial, Helvetica, sans-serif;
          font-weight:1000;
          }

          .tableHeader
          {
          font-size: small;
          font-family:Arial, Helvetica, sans-serif;
          font-weight:1000;
          }

          .divimage
          {
          position:absolute;
          top: 0;
          left: 0;
          z-index:2;
          }
          .divimage2
          {
          position:absolute;
          top: 0;
          left: 0;
          z-index:1;
          }

          .copyright{
          font-size: 8pt;
          font-family:Arial, Helvetica, sans-serif;
          text-align: center;
          }
        </style>
				<META NAME="author" CONTENT="Cyberoptics Corporation"></META>
				<META NAME="description" CONTENT="Project Manifest Stylesheet"></META> 
			</head>
			<body style="font-family: Arial" bgcolor="#CCCCFF">
        <div class="divimage">
          <img alt="Logo" src="images\cyberoptics.bmp"/>
        </div>
        <img position="absolute" top="0" alt="border" src="images\border.bmp" width="100%" height="41"/>
        <xsl:apply-templates select="project"/>
				<br></br>
				<p class="copyright">Copyright 2010 Cyberoptics Corporation</p>
			</body>
		</html>
	</xsl:template>

  <xsl:template match="project">
    <h2>
      <xsl:value-of select="@name"></xsl:value-of> System Software Version 
      <xsl:value-of select="@version"></xsl:value-of>.<xsl:value-of select="@build_number"></xsl:value-of>
      Dependency Report
    </h2>

    <h3>Software Dependencies</h3>
    <xsl:apply-templates select="dependencies"/>

    <h3>Hardware Dependencies</h3>
    <xsl:apply-templates select="hardware_dependencies"/>

  </xsl:template>

  <xsl:template match="dependencies">
    <h4>
      <table width="100%">
        <tr>
          <td class="tableHeader" width="20%">
            <b>Dependency Name</b>
          </td>
          <td width="3%"></td>
          <td class="tableHeader" width="10%">
            <b>Version</b>
          </td>
          <td width="3%"></td>
          <td class="tableHeader" width="64%">
            <b>Source</b>
          </td>
        </tr>
        <xsl:apply-templates select="dependency"/>
      </table>
    </h4>
  </xsl:template>
	
	<xsl:template match="dependency">
    <tr>
      <td>
        <xsl:value-of select="@project_name"></xsl:value-of>
      </td>
      <td></td>
      <td>
        <xsl:value-of select="@version"></xsl:value-of>
      </td>
      <td width="3%"></td>
      <td>
        <xsl:value-of select="@source"></xsl:value-of>
      </td>
    </tr>
  </xsl:template>

  <xsl:template match="hardware_dependencies">
    <h4>
      <table width="100%">
        <tr>
          <td class="tableHeader" width="20%">
            <b>Dependency Name</b>
          </td>
          <td width="3%"></td>
          <td class="tableHeader" width="77%">
            <b>Version</b>
          </td>
        </tr>
        <xsl:apply-templates select="hardware_dependency"/>
      </table>
    </h4>
    <br></br>
  </xsl:template>

  <xsl:template match="hardware_dependency">
    <tr>
      <td>
        <xsl:value-of select="@name"></xsl:value-of>
      </td>
      <td></td>
      <td>
        <xsl:value-of select="@version"></xsl:value-of>
      </td>
      <td width="3%"></td>
    </tr>
  </xsl:template>
 
</xsl:stylesheet>
