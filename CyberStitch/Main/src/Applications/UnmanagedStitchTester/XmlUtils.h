#pragma once

#import <MSXML6.dll>
#include <string>

using namespace MSXML2;

typedef MSXML2::IXMLDOMNodePtr XmlNode;
typedef MSXML2::IXMLDOMNodeListPtr XmlNodeList;
typedef MSXML2::IXMLDOMDocument2Ptr XmlDOM;
typedef MSXML2::IXMLDOMElementPtr XmlElement;

class XmlUtils
{
public:
	static bool GetBoolAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, bool defaultVal=false);
	static int GetIntAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, int defaultVal=-1);
	static double GetDblAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, double defaultVal=-1.0);
	static _bstr_t GetStringAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, _bstr_t defaultVal="");

	static int GetIntElement(std::string elementName, XmlNode pNode, int defaultVal=-1);
	static double GetDblElement(std::string elementName, XmlNode pNode, double defaultVal=-1.0);
	static _bstr_t GetStringElement(std::string elementName, XmlNode pNode, _bstr_t defaultVal="");
	static void SetIntElement(std::string elementName, XmlNode pNode, XmlDOM pDOM, int Value);
	static void SetDblElement(std::string elementName, XmlNode pNode, XmlDOM pDOM, double Value);
	static void SetStringElement(std::string elementName, XmlNode pNode, XmlDOM pDOM, _bstr_t Value);
};
