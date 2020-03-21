// =============================================================================
// Import input data for simulation
//
// =============================================================================
// Copyright 2018 <University of Washington, LEMS>
// Hiromi Yasuda (2018/08/11)
// =============================================================================

#ifndef READINPUTDATA_H
#define READINPUTDATA_H

#include <iostream>

class ReadInputData {
  // accessor
 public:
  std::string sDIR_data;
  int n_width;
  int n_height;

  int importFile(std::string filename);
  inline double GetReal(const std::string &s);
  inline int GetInt(const std::string &s);
  inline std::string GetString(const std::string &s);
};
#endif // READINPUTDATA_H
