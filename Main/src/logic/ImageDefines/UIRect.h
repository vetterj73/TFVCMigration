#pragma once

/*
	A Rectangle in Unsinged int
*/

class UIRect
{
public: 
	UIRect();
	UIRect(
		unsigned int firstCol,
		unsigned int lastCol,
		unsigned int firstRow,
		unsigned int lastRow );

	UIRect(const UIRect& b);

	void operator=(const UIRect& b);

	unsigned int FirstColumn;
	unsigned int LastColumn;
	unsigned int FirstRow;
	unsigned int LastRow;

	bool IsValid() const;

	unsigned int Rows() const;

	unsigned int Columns() const;
};



