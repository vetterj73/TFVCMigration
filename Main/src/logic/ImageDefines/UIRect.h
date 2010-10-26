#pragma once

class UIRect
{
public: 
	UIRect();
	UIRect(
		unsigned int firstCol,
		unsigned int lastCol,
		unsigned int firstRow,
		unsigned int lastRow ) 
	{
		FirstColumn	= firstCol;
		LastColumn	= lastCol;
		FirstRow	= firstRow;
		LastRow		= lastRow;
	}

	unsigned int FirstColumn;
	unsigned int LastColumn;
	unsigned int FirstRow;
	unsigned int LastRow;

	bool IsValid() const
	{
		return FirstColumn<LastColumn && 
				FirstRow<LastRow;
	}

	unsigned int Rows() const
	{
		if( !IsValid() )
			return 0;
		return LastRow-FirstRow+1; 
	}

	unsigned int Columns() const
	{
		if( !IsValid() )
			return 0;

		return LastColumn-FirstColumn+1; 
	}



