// =============================================================================
// OpenCV: Example of motion tracking (OpticalFlowFarneback)
//
// =============================================================================
// Copyright 2018 <University of Washington, LEMS>
// Authors: Hiromi Yasuda (2018/10/09)
// =============================================================================
//
// =============================================================================

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "ReadInputData.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctype.h>

int main(int argc, char** argv) {
  // show help
  if (argc < 2) {
    std::cout<<
        " Usage: ./opencv_exe <video_name>\n"
        " examples:\n"
        " ./opencv_exe Movies/Chronos_demo.mp4\n"
             << std::endl;
    return 0;
  }
  std::cout << "  step 1: read inputdata.data" << std::endl;
  ReadInputData indata;
  indata.importFile("./Inputdata.dat");

  // Set input video
  std::string video = argv[1];
  cv::VideoCapture cap(video);

  // Get the information of video
  double fps = cap.get(CV_CAP_PROP_FPS);
  std::stringstream fps_info;
  fps_info << "FPS = " << roundf(fps);

  // Get image size
  int Width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  int Height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  int max_frame = cap.get(CV_CAP_PROP_FRAME_COUNT);
  std::cout << "Maximum frame number = " << max_frame << std::endl;
  cv::Mat source(Height, Width, CV_8UC1);
  cv::Mat HIS_source(Height, Width, CV_8UC1);

  // Export movie file
  cv::VideoWriter writer("output.mp4",
                         // cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),  // AVI
                         cv::VideoWriter::fourcc('H', '2', '6', '4'),  // MP4
                         roundf(fps), cv::Size(Width, Height), true);
  // Analysis
  for (int i_frame=0; i_frame < max_frame; i_frame++) {
    cap >> source;
    cv::Mat disp = source.clone();
    cv::cvtColor(source, source, CV_BGR2GRAY);

    if (i_frame > 0) {
      std::vector<cv::Point2f> prev_pts;
      std::vector<cv::Point2f> next_pts;

      cv::Size flowSize(indata.n_width, indata.n_height);  // Number of vectors
      cv::Point2f center = cv::Point(source.cols/2., source.rows/2.);
      for (int i=0; i < flowSize.width; ++i) {
        for (int j=0; j < flowSize.height; ++j) {
          cv::Point2f p(i*float(source.cols)/(flowSize.width-1),
                        j*float(source.rows)/(flowSize.height-1));
          prev_pts.push_back((p-center)*0.95f+center);
        }
      }

      cv::Mat flow;
      std::vector<float> error;

      calcOpticalFlowFarneback(HIS_source, source, flow, 0.8, 10, 15, 3, 5, 1.1, 0);

      // Display and Save optical flow
      std::ostringstream oss;
      oss << std::setfill('0') << std::setw(3) << i_frame;
      std::ofstream fs1(indata.sDIR_data+"/opticalflow_"+oss.str()+".csv");

      std::vector<cv::Point2f>::const_iterator p = prev_pts.begin();
      for (; p != prev_pts.end(); ++p) {
        const cv::Point2f& fxy = flow.at<cv::Point2f>(p->y, p->x);
        cv::line(disp, *p, *p+fxy*8, cv::Scalar(255, 0, 0), 1);
        // Save data
        double magnitude = sqrt(fxy.x*fxy.x + fxy.y*fxy.y);
        fs1 << p->y << ", " << p->x << ", " << magnitude << ", " << fxy.x << ", " << fxy.y << std::endl;
      }
      HIS_source = source.clone();
      // Display the frame
      imshow("vector", disp);

      // Write the frame into the movie file
      writer << disp;

      int c = cv::waitKey(1);
      if (c == 27) return 0;
    }

    std::cout << "Frame number = " << i_frame << std::endl;
  }
  cv::waitKey(0);
  return 0;
}

