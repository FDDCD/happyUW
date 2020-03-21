// =============================================================================
// Import input data for simulation
//
// =============================================================================
// Copyright 2018 <University of Washington, LEMS>
// Authors: Hiromi Yasuda (2018/08/16)
// =============================================================================


#include "ReadInputData.h"

#include <iostream>
#include <cstdio>
#include <cmath>

#include <string>
#include <sstream>
#include <fstream>

// =============================================================================
// Import input data file
// =============================================================================
int ReadInputData::importFile(std::string filename) {
  std::ifstream file;
  file.open(filename);
  if (!file || !file.is_open() || file.bad() || file.fail()) {
    std::cout << "Error! File cannot be opened." << std::endl;
    return 1;
  }
  std::string buf;
  std::string::size_type comment_start = 0;
  // std::vector< ChVector<> > vrtx_geo;

  // Analyze line by line -----------------------
  while (getline(file, buf)) {
      // Use '//' as a comment line (Ignore characters after '//')
      if ( (comment_start = buf.find("//")) != std::string::size_type(-1) )
        buf = buf.substr(0, comment_start);
      // Ignore empty lines
      if (buf.empty())
        continue;

      // Extract information
      size_t pos;
      // Extract input information ------------------------------------
      if ((pos = buf.find("sDIR_data")) != std::string::npos) {
        sDIR_data = GetString(buf);
        std::cout << "    sDIR_data  = " << sDIR_data << std::endl;

      } else if ((pos = buf.find("n_width")) != std::string::npos) {
        n_width = GetInt(buf);
        std::cout << "    n_width    = " << n_width << std::endl;

      } else if ((pos = buf.find("n_height")) != std::string::npos) {
        n_height = GetInt(buf);
        std::cout << "    n_height   = " << n_height << std::endl;
      }  // (Extract info end)
    }  // (while end)
  file.close();
  std::cout << "Finish importing Inputdata file!" << std::endl;

  return 0;
}

// Extract Real or integer numbers ============================================
inline double ReadInputData::GetReal(const std::string &s) {
    size_t pos;
    double dataReal = 0.0;

    if ((pos = s.find('=')) != std::string::npos) {
      while (pos != std::string::npos) {
            size_t pos1 = pos;
            if ((pos1 = s.find(';', pos+1)) != std::string::npos) {
              dataReal = std::atof(s.substr(pos+1, (pos1-(pos+1))).c_str());
              break;
            } else {
                break;
            }
      }  // (while end)
    }  // (if end)
    return dataReal;
}

inline int ReadInputData::GetInt(const std::string &s) {
    size_t pos;
    int dataInt = 0;

    if ((pos = s.find('=')) != std::string::npos) {
      while (pos != std::string::npos) {
        size_t pos1 = pos;
        if ((pos1 = s.find(';', pos+1)) != std::string::npos) {
          dataInt = std::atof(s.substr(pos+1, (pos1-(pos+1))).c_str());
          break;
        } else {
          break;
        }
      }  // (while end)
    }  // (if end)
    return dataInt;
}

inline std::string ReadInputData::GetString(const std::string &s) {
    size_t pos;
    std::string dataString = "aaa";

    if ((pos = s.find('=')) != std::string::npos) {
      while (pos != std::string::npos) {
        size_t pos1 = pos;
        if ((pos1 = s.find(';', pos+1)) != std::string::npos) {
          dataString = s.substr(pos+1, (pos1-(pos+1))).c_str();
          break;
        } else {
          break;
        }
      }  // (while end)
    }  // (if end)

    // Remove double quotations "
    for (size_t c = dataString.find_first_of('"');
         c != std::string::npos; c = c = dataString.find_first_of('"')) {
      dataString.erase(c, 1);
    }
    // Remove space
    for (size_t c = dataString.find_first_of(' ');
         c != std::string::npos; c = c = dataString.find_first_of(' ')) {
      dataString.erase(c, 1);
    }
    return dataString;
}
