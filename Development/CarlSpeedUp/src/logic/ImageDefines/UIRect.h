#pragma once

/*
	A Rectangle in Unsigned int
*/

class UIRect
{
public: 
	UIRect();
	UIRect(
		unsigned int firstCol,
		unsigned int firstRow,
		unsigned int lastCol,		
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

	double RowCenter() const;
	double ColumnCenter() const;
	
};

/*
	A Rectangle in double
*/
struct DRect
{
	double xMin;
	double xMax;
	double yMin;
	double yMax;

	DRect();
	void Reset();
	
	double Width() const;
	double Height() const;
	double Area() const;
	double CenX() const;
	double CenY() const;

	bool IsValid() const;

	DRect OverlapRect(DRect inRect) const;
};




