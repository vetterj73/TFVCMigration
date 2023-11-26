#pragma once

#include "regoff.h"

namespace RegoffTester {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for Form1
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::Button^  bTBrowser;
	protected: 
	private: System::Windows::Forms::Button^  bTRun;
	private: System::Windows::Forms::TextBox^  tBFile;
	private: System::Windows::Forms::OpenFileDialog^  openFileDialog1;

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->bTBrowser = (gcnew System::Windows::Forms::Button());
			this->bTRun = (gcnew System::Windows::Forms::Button());
			this->tBFile = (gcnew System::Windows::Forms::TextBox());
			this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			this->SuspendLayout();
			// 
			// bTBrowser
			// 
			this->bTBrowser->Location = System::Drawing::Point(247, 25);
			this->bTBrowser->Name = L"bTBrowser";
			this->bTBrowser->Size = System::Drawing::Size(33, 23);
			this->bTBrowser->TabIndex = 0;
			this->bTBrowser->Text = L"...";
			this->bTBrowser->UseVisualStyleBackColor = true;
			this->bTBrowser->Click += gcnew System::EventHandler(this, &Form1::bTBrowser_Click);
			// 
			// bTRun
			// 
			this->bTRun->Location = System::Drawing::Point(105, 73);
			this->bTRun->Name = L"bTRun";
			this->bTRun->Size = System::Drawing::Size(75, 23);
			this->bTRun->TabIndex = 1;
			this->bTRun->Text = L"Run";
			this->bTRun->UseVisualStyleBackColor = true;
			this->bTRun->Click += gcnew System::EventHandler(this, &Form1::bTRun_Click);
			// 
			// tBFile
			// 
			this->tBFile->Location = System::Drawing::Point(12, 25);
			this->tBFile->Name = L"tBFile";
			this->tBFile->Size = System::Drawing::Size(229, 20);
			this->tBFile->TabIndex = 2;
			// 
			// openFileDialog1
			// 
			this->openFileDialog1->FileName = L"openFileDialog1";
			// 
			// Form1
			// 
			this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->ClientSize = System::Drawing::Size(292, 122);
			this->Controls->Add(this->tBFile);
			this->Controls->Add(this->bTRun);
			this->Controls->Add(this->bTBrowser);
			this->Name = L"Form1";
			this->Text = L"RegoffTester";
			this->ResumeLayout(false);
			this->PerformLayout();

		}
#pragma endregion

		private: System::Void bTBrowser_Click(System::Object^  sender, System::EventArgs^  e) 
		{
			if(openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK)
			{
				tBFile->Text = openFileDialog1->FileName;
			}
		}
		private: System::Void bTRun_Click(System::Object^  sender, System::EventArgs^  e) 
		{
			//control paramters
			int decim = 4;
			bool bApplyUpLimit = true;
			int iMaxCols = 1024;
			int iMaxRows = 1024;

			// Read image
			Bitmap bmp(openFileDialog1->FileName);

			int iWidth = bmp.Width;
			int iHeight = bmp.Height;

			unsigned char* pcRedBuf = new unsigned char[iWidth*iHeight];
			unsigned char* pcGreenBuf = new unsigned char[iWidth*iHeight];

			int ix, iy;
			Color color;
			for(iy=0; iy<iHeight; iy++)
			{
				for(ix=0; ix<iWidth; ix++)
				{
					color = bmp.GetPixel(ix,iy);
					pcRedBuf[iy*iWidth+ix] = color.R;
					pcGreenBuf[iy*iWidth+ix] = color.G;
				}
			}

			int iOffsetX=0, iOffsetY=0;
			int iRWidth = iWidth, iRHeight = iHeight;
			if(bApplyUpLimit  && iWidth>iMaxCols && iHeight>iMaxRows)
			{
				iRWidth = 1024;
				iRHeight = 1024;
				iOffsetX = (iWidth - iRWidth)/2;
				iOffsetY = (iHeight - iRHeight)/2;
			}

			//Regoff paremeter		
			complexf *z(0);
			REGLIM  *lims(0);   /* Limits of search range for registration offset.  Use
						   if there is _a priori_ knowledge about the offset.
						   If a null pointer is passed, the search range
						   defaults to the entire range of values

							  x = [-ncols/2, ncols/2>
							  y = [-nrows/2, nrows/2>

						   Excessively large values are clipped to the above
						   range; inconsistent values result in an error return.
						*/

			int      job(2);   // Allow negative correlation

			int      histclip(1);   /* Histogram clipping factor; clips peaks to prevent
								   noise in large flat regions from being excessively
								   amplified.  Use histclip=1 for no clipping;
								   histclip>1 for clipping.  Recommended value = 32 */

			int      dump(1);       /* Dump intermediate images to TGA files
								   (useful for debugging):

								   ZR.TGA and ZI.TGA are decimated (and possibly
									  histogram-equalized images that are input to the
									  correlation routine.

								   PCORR.TGA is the correlogram.

								   HOLE.TGA is the correlogram, excluding the vicinity
									  of the peak. */

			double dOffsetX=0, dOffsetY=0, dCorScore=-2, dAmbig=-2;

			char myChar[512];
			char** myCharPtr = (char**)(&myChar);

			int nrows = RegoffLength(iRHeight, decim);
			int ncols = RegoffLength(iRWidth, decim);

			// Regoff
			int iFlag =regoff(ncols, nrows, 
				pcRedBuf+iOffsetY*iWidth+iOffsetX, 
				pcGreenBuf+iOffsetY*iWidth+iOffsetX, 
				iWidth, iWidth, z, decim, decim, lims, job, 
				histclip, dump, &dOffsetX, &dOffsetY,
				&dCorScore, &dAmbig, myCharPtr/*&error_msg*/);

			if(iFlag == 0)
			{
				// drain 
				int iLocX, iLocY, iValue;
				for(iy=0; iy<iHeight; iy++)
				{
					for(ix=0; ix<iWidth; ix++)
					{
						iLocX = ix+(int)dOffsetX;
						iLocY = iy+(int)dOffsetY;
						if(iLocX<0 || iLocX>=iWidth || iLocY<0 ||iLocY>=iHeight)
							iValue = 0;
						else
							iValue = pcGreenBuf[iLocY*iWidth+iLocX];
						
						color = Color::FromArgb(pcRedBuf[iy*iWidth+ix], iValue, 0);

						bmp.SetPixel(ix, iy, color);
					}
				}

				String^ fileName = System::IO::Path::GetFileName(openFileDialog1->FileName);
				fileName = "C:\\Temp\\"+fileName;
				bmp.Save(fileName);			
				
				// report
				String^ sResult = String::Format("Xoffset={0}, yOffset={1}, Score={2}, Ambig={3}", 
					dOffsetX, dOffsetY, dCorScore, dAmbig);

				System::Windows::Forms::MessageBox::Show(sResult, "Done");
			}
			else
			{
				String^ sResult = String::Format("Error case {0}", iFlag);
				System::Windows::Forms::MessageBox::Show(sResult, "Failed!");
			}
		}
	};
}

