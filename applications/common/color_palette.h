#pragma once

#include <map>
#include <vector>

typedef std::vector< std::vector<double> > Palette;

/////////////////////////////////////////////////
enum PaletteType {
  eOriginalPalette = 1,
  eNewPalette      = 2,
  eOblivion        = 3
};

//////////////////////////////////////////////////
class ColorPalette {

 public:

  ColorPalette() {
    _InitColorPalettes();
  }
  ~ColorPalette() {

  }

  Palette  GetPalette( PaletteType type ) {
    std::map< PaletteType, Palette >::iterator it = m_mPalettes.find(type);
    if( it == m_mPalettes.end() ) {
      std::cerr << "Error: Color palete not recognized" << std::endl;
      //        exit(1);
    }
    return m_mPalettes[type];
  }

  Palette& GetPaletteRef( PaletteType type ) {

    std::map< PaletteType, Palette >::iterator it = m_mPalettes.find(type);
    if( it == m_mPalettes.end() ) {
      std::cerr << "Error: Color palete not recognized" << std::endl;
      //        exit(1);
    }
    return m_mPalettes[type];
  }

 private:
  void _InitColorPalettes() {
    Palette p;
    std::vector<double> vColor(3);

    ///////////////////////////////////////////////////////////////////////////
    // ORIGINAL
    ///////////////////////////////////////////////////////////////////////////
    vColor[0] = 1.0;  vColor[1] = 1.0;  vColor[2] = 1.0;  p.push_back(vColor); // white
    vColor[0] = 0.2;  vColor[1] = 0.47; vColor[2] = 1.0;  p.push_back(vColor); // light blue
    vColor[0] = 0.9;  vColor[1] = 0.9;  vColor[2] = 0.3;  p.push_back(vColor); // ocre
    vColor[0] = 0.6;  vColor[1] = 0.0;  vColor[2] = 0.6;  p.push_back(vColor); // magenta
    vColor[0] = 0.7;  vColor[1] = 0.7;  vColor[2] = 0.7;  p.push_back(vColor); // light gray
    vColor[0] = 0.4;  vColor[1] = 0.0;  vColor[2] = 0.9;  p.push_back(vColor); // medium blue
    vColor[0] = 0.7;  vColor[1] = 0.7;  vColor[2] = 0.7;  p.push_back(vColor); // light gray
    vColor[0] = 0.6;  vColor[1] = 0.1;  vColor[2] = 0.1;  p.push_back(vColor); // dark red
    vColor[0] = 0.7;  vColor[1] = 0.7;  vColor[2] = 0.3;  p.push_back(vColor); // ocre
    vColor[0] = 0.6;  vColor[1] = 0.0;  vColor[2] = 0.8;  p.push_back(vColor); // magenta
    vColor[0] = 0.2;  vColor[1] = 0.47; vColor[2] = 1.0;  p.push_back(vColor); // light blue
    vColor[0] = 0.4;  vColor[1] = 0.0;  vColor[2] = 0.9;  p.push_back(vColor); // medium blue
    vColor[0] = 0.6;  vColor[1] = 0.0;  vColor[2] = 0.8;  p.push_back(vColor); // magenta
    vColor[0] = 0.7;  vColor[1] = 0.7;  vColor[2] = 0.7;  p.push_back(vColor); // light gray
    vColor[0] = 0.7;  vColor[1] = 0.7;  vColor[2] = 0.7;  p.push_back(vColor); // light gray
    vColor[0] = 0.1;  vColor[1] = 0.9;  vColor[2] = 0.5;  p.push_back(vColor); // randomish

    m_mPalettes[eOriginalPalette] = p;

    ///////////////////////////////////////////////////////////////////////////
    // TEST
    ///////////////////////////////////////////////////////////////////////////
    p.clear();
    vColor[0] = 128.0/255.0; vColor[1] = 153.0/255.0; vColor[2] = 148.0/255.0;  p.push_back(vColor);
    vColor[0] = 174.0/255.0; vColor[1] = 204.0/255.0; vColor[2] = 182.0/255.0;  p.push_back(vColor);
    vColor[0] = 222.0/255.0; vColor[1] = 242.0/255.0; vColor[2] = 196.0/255.0;  p.push_back(vColor);
    vColor[0] = 229.0/255.0; vColor[1] = 104.0/255.0; vColor[2] =  63.0/255.0;  p.push_back(vColor);
    vColor[0] = 128.0/255.0; vColor[1] = 153.0/255.0; vColor[2] = 148.0/255.0;  p.push_back(vColor);
    vColor[0] = 174.0/255.0; vColor[1] = 204.0/255.0; vColor[2] = 182.0/255.0;  p.push_back(vColor);
    vColor[0] = 222.0/255.0; vColor[1] = 242.0/255.0; vColor[2] = 196.0/255.0;  p.push_back(vColor);
    vColor[0] = 229.0/255.0; vColor[1] = 104.0/255.0; vColor[2] =  63.0/255.0;  p.push_back(vColor);
    vColor[0] = 128.0/255.0; vColor[1] = 153.0/255.0; vColor[2] = 148.0/255.0;  p.push_back(vColor);
    vColor[0] = 174.0/255.0; vColor[1] = 204.0/255.0; vColor[2] = 182.0/255.0;  p.push_back(vColor);
    vColor[0] = 222.0/255.0; vColor[1] = 242.0/255.0; vColor[2] = 196.0/255.0;  p.push_back(vColor);
    vColor[0] = 229.0/255.0; vColor[1] = 104.0/255.0; vColor[2] =  63.0/255.0;  p.push_back(vColor);
    vColor[0] = 128.0/255.0; vColor[1] = 153.0/255.0; vColor[2] = 148.0/255.0;  p.push_back(vColor);
    vColor[0] = 174.0/255.0; vColor[1] = 204.0/255.0; vColor[2] = 182.0/255.0;  p.push_back(vColor);
    vColor[0] = 222.0/255.0; vColor[1] = 242.0/255.0; vColor[2] = 196.0/255.0;  p.push_back(vColor);
    vColor[0] = 229.0/255.0; vColor[1] = 104.0/255.0; vColor[2] =  63.0/255.0;  p.push_back(vColor);

    m_mPalettes[eNewPalette] = p;

    p.clear();
    vColor[0] = 104.0/255.0; vColor[1] = 141.0/255.0; vColor[2] = 150.0/255.0;  p.push_back(vColor);
    vColor[0] = 103.0/255.0; vColor[1] = 196.0/255.0; vColor[2] = 213.0/255.0;  p.push_back(vColor);
    vColor[0] = 207.0/255.0; vColor[1] = 253.0/255.0; vColor[2] = 250.0/255.0;  p.push_back(vColor);
    vColor[0] = 255.0/255.0; vColor[1] = 254.0/255.0; vColor[2] = 210.0/255.0;  p.push_back(vColor);
    vColor[0] = 255.0/255.0; vColor[1] = 106.0/255.0; vColor[2] = 74.0/255.0;   p.push_back(vColor);

    m_mPalettes[eOblivion] = p;
  }

 private:
  std::map< PaletteType, Palette > m_mPalettes;

};
