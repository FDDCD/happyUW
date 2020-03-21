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
#include <opencv2/opencv.hpp>
#include <opencv2/superres/optical_flow.hpp>

#include <iostream>
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
  std::cout << "width = " << Width << ", height = " << Height << std::endl;
  // Export movie file
  cv::VideoWriter writer("output.mp4",
                         // cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),  // AVI
                         cv::VideoWriter::fourcc('H', '2', '6', '4'),  // MP4
                         roundf(fps), cv::Size(Width, Height), true);

  // Opticalflow Algorithm
  cv::Ptr<cv::superres::DenseOpticalFlowExt> opticalFlow = cv::superres::createOptFlow_Farneback();
  // Save previous frame
  cv::Mat prev;
  cap >> prev;

  for (int i_frame=0; i_frame < max_frame; i_frame++) {
    cv::Mat curr;
    cap >> curr;

    // Calculate optical flow
    cv::Mat flowX, flowY;
    opticalFlow->calc(prev, curr, flowX, flowY);

    // Visualize opticalflow
    // Use Polar coordinate (degrees)
    cv::Mat magnitude, angle;
    cartToPolar(flowX, flowY, magnitude, angle, true);
    // Hue: Angle of optical flow
    // Saturation: Normalized length of optical flow
    // Value: Always 1
    cv::Mat hsvPlanes[3];
    hsvPlanes[0] = angle;
    normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);  // Normalize
    hsvPlanes[1] = magnitude;
    hsvPlanes[2] = cv::Mat::ones(magnitude.size(), CV_32F);
    // Merge to one frame
    cv::Mat hsv;
    merge(hsvPlanes, 3, hsv);
    //  Convert HSV to BGR
    cv::Mat flowBgr;
    cv::cvtColor(hsv, flowBgr, cv::COLOR_HSV2BGR);
    // std::cout << "width = " << flowBgr.cols << ", height = " << flowBgr.rows << std::endl;
    // Display
    cv::imshow("input", curr);
    cv::imshow("optical flow", flowBgr);

    // Renew prev
    prev = curr;

    // Write the frame into the movie file
    // cv::Mat image8Bit;
    // flowBgr.convertTo(image8Bit, CV_8UC3);
    // writer << image8Bit;
    // cv::imshow("image8Bit", image8Bit);
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(3) << i_frame;
    cv::imwrite("Result_opencv/out_"+oss.str()+".tiff", flowBgr);

    int c = cv::waitKey(1);
    if (c == 27) return 0;
  }
  cv::waitKey(0);
  return 0;
}


