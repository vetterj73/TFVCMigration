#include "stdafx.h"

#include "XmlUtils.h"


	int XmlUtils::GetIntAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, int defaultVal)
	{
		int value;
		if(pElement == NULL || pElement->getAttributeNode(attrName.c_str()) == NULL)
			return defaultVal;

		_variant_t tmp = pElement->getAttributeNode(attrName.c_str())->value;

		try
		{
			value = int(tmp);
		}
		catch (...)
		{
			value = defaultVal;
		}
		return value;
	}
	
	_bstr_t XmlUtils::GetStringAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, _bstr_t defaultVal)
	{
		if(pElement == NULL || pElement->getAttributeNode(attrName.c_str()) == NULL)
			return defaultVal;

		_bstr_t tmp = pElement->getAttributeNode(attrName.c_str())->value;

		return tmp;
	}

	double XmlUtils::GetDblAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, double defaultVal)
	{
		if(pElement == NULL || pElement->getAttributeNode(attrName.c_str()) == NULL)
			return defaultVal;

		HRESULT hr;
		std::string str;
		_variant_t varValue = defaultVal;

		hr = ::VariantChangeTypeEx(&varValue, &varValue, LOCALE_INVARIANT, 0, VT_BSTR);
		if (SUCCEEDED(hr))
		{
			varValue = pElement->getAttribute(attrName.c_str());
			if (varValue.vt != VT_NULL)
			{
				str = (_bstr_t)varValue;

				double d = atof(str.c_str());
				return d;
			}
		}

		return defaultVal;
	}

	bool XmlUtils::GetBoolAttr(std::string attrName, MSXML2::IXMLDOMElementPtr pElement, bool defaultVal)
	{
		bool value;
		if(pElement == NULL || pElement->getAttributeNode(attrName.c_str()) == NULL)
			return defaultVal;

		_variant_t tmp = pElement->getAttributeNode(attrName.c_str())->value;

		try
		{
			value = bool(tmp);
		}
		catch (...)
		{
			value = defaultVal;
		}
		return value;
	}
	
	int XmlUtils::GetIntElement(std::string elementName, XmlNode pNode, int defaultVal)
	{
		int value;
		if(pNode == NULL) return defaultVal;

		MSXML2::IXMLDOMElementPtr pElement = pNode->selectSingleNode(elementName.c_str());
		if(pElement == NULL) return defaultVal;

		_variant_t tmp = pElement->getAttributeNode("Value")->value;

		try
		{
			value = int(tmp);
		}
		catch (...)
		{
			value = defaultVal;
		}
		return value;
	}
	
	double XmlUtils::GetDblElement(std::string elementName, XmlNode pNode, double defaultVal)
	{
		if(pNode == NULL) return defaultVal;

		MSXML2::IXMLDOMElementPtr pElement = pNode->selectSingleNode(elementName.c_str());
		if(pElement == NULL) return defaultVal;

		HRESULT hr;
		std::string str;
		_variant_t varValue = defaultVal;

		hr = ::VariantChangeTypeEx(&varValue, &varValue, LOCALE_INVARIANT, 0, VT_BSTR);
		if (SUCCEEDED(hr))
		{
			varValue = pElement->getAttribute("Value");
			if (varValue.vt != VT_NULL)
			{
				str = (_bstr_t)varValue;

				double d = atof(str.c_str());
				return d;
			}
		}

		return defaultVal;
	}

	_bstr_t XmlUtils::GetStringElement(std::string elementName, XmlNode pNode, _bstr_t defaultVal)
	{
		if(pNode == NULL) return defaultVal;

		MSXML2::IXMLDOMElementPtr pElement = pNode->selectSingleNode(elementName.c_str());
		if(pElement == NULL) return defaultVal;

		MSXML2::IXMLDOMAttributePtr pAttr = pElement->getAttributeNode("Value");
		if(pAttr == NULL) return defaultVal;

		_bstr_t tmp = pAttr->value;

		return tmp;
	}

	void XmlUtils::SetIntElement(std::string elementName, XmlNode pNode, XmlDOM pDOM, int Value)
	{
		if(pNode == NULL) return;

		MSXML2::IXMLDOMElementPtr pElement = pNode->selectSingleNode(elementName.c_str());
		if(pElement == NULL)
		{
			pElement = pDOM->createElement(elementName.c_str());
			pNode->appendChild(pElement);
			pElement = pNode->selectSingleNode(elementName.c_str());
		}
		if(pElement != NULL)
		{
			pElement->setAttribute("Value", Value);
		}
	}
	
	void XmlUtils::SetDblElement(std::string elementName, XmlNode pNode, XmlDOM pDOM, double Value)
	{
		if(pNode == NULL) return;

		MSXML2::IXMLDOMElementPtr pElement = pNode->selectSingleNode(elementName.c_str());
		if(pElement == NULL)
		{
			pElement = pDOM->createElement(elementName.c_str());
			pNode->appendChild(pElement);
			pElement = pNode->selectSingleNode(elementName.c_str());
		}
		if(pElement != NULL)
		{
			HRESULT hr;
			_variant_t varValue = Value;
			hr = ::VariantChangeTypeEx(&varValue, &varValue, LOCALE_INVARIANT, 0, VT_BSTR);
			hr = pElement->setAttribute("Value", varValue);
		}
	}
	
	void XmlUtils::SetStringElement(std::string elementName, XmlNode pNode, XmlDOM pDOM, _bstr_t Value)
	{
		if(pNode == NULL) return;

		MSXML2::IXMLDOMElementPtr pElement = pNode->selectSingleNode(elementName.c_str());
		if(pElement == NULL)
		{
			pElement = pDOM->createElement(elementName.c_str());
			pNode->appendChild(pElement);
			pElement = pNode->selectSingleNode(elementName.c_str());
		}
		if(pElement != NULL)
		{
			pElement->setAttribute("Value", Value);
		}
	}
	
