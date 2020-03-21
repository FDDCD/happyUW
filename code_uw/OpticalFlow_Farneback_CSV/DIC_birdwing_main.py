import numpy as np
import matplotlib.pyplot as plt
import time

import os
import subprocess
import platform
import shutil   #High level file operation
import glob

 
# Figure parameters =================================================
# When you insert the figure, you need to make fig height 2
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['figure.figsize']  = 8, 6      # (w=3,h=2) multiply by 3
plt.rcParams['font.size']       = 24        # Original fornt size is 8, multipy by above number
#plt.rcParams['text.usetex']     = True
#plt.rcParams['ps.useafm'] = True
#plt.rcParams['pdf.use14corefonts'] = True
#plt.rcParams['text.latex.preamble'] = '\usepackage{sfmath}'
plt.rcParams['lines.linewidth'] = 3.   
plt.rcParams['lines.markersize'] = 8. 
plt.rcParams['legend.fontsize'] = 20        # Original fornt size is 8, multipy by above number
plt.rcParams['xtick.labelsize'] = 24        # Original fornt size is 8, multipy by above number
plt.rcParams['ytick.labelsize'] = 24        # Original fornt size is 8, multipy by above number
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams['figure.subplot.left']  = 0.2
#plt.rcParams['figure.subplot.right']  = 0.7
plt.rcParams['figure.subplot.bottom']  = 0.2
plt.rcParams['savefig.dpi']  = 300


# =============================================================================
# Reinforcement learning for origami
# =============================================================================
class DIC_Analysis:
    def __init__(self, video_name, dir_result):
        self.video_name = video_name
        self.dir_result = dir_result
        

    def CreateInputdata(self, n_width, n_height):
        print('Start creating Input file')
        #Export text file
        with open('Inputdata.dat','w') as f:
            f.write('//*********************************************************************\n')
            f.write('//*  Input data for DIC Analysis (based on OpenCV)                    *\n')
            f.write('//*********************************************************************\n')
#            f.write('sVideo     = "{}"; // File name of the movie \n'.format(self.video_name))
            f.write('sDIR_data  = "{}"; // Directory for the data save\n'.format(self.dir_result))
            f.write('n_width    = {}; // Number of subdivision (width) \n'.format(n_width))
            f.write('n_height   = {}; // Number of subdivision (height) \n'.format(n_height))
        return()
    
    def DIC_OpticalFlow(self, n_width, n_height):
        # Create input data for Project Chrono
        self.CreateInputdata(n_width, n_height)
        
        
        # Delete previous data
        if os.path.exists(self.dir_result):
            shutil.rmtree(self.dir_result)
        os.makedirs(self.dir_result)

        
        # Run opencv_exe
        if platform.system()=="Darwin":
            subprocess.call('./opencv_exe {}'.format(self.video_name), shell=True)
#            proc = subprocess.Popen(["./opencv_exe {}".format(self.video_name)],
#                                     stdout=subprocess.PIPE, stdin=subprocess.PIPE)
#            # Receive the rest of the data
#            stdout_data, stderr_data = proc.communicate()
#            
#            for i in range(200):
#                result = proc.stdout.readline().strip()
#                print(str(result, 'utf-8'))
        else:
            subprocess.call('./opencv_exe {}'.format(video_name), shell=True)
        return()
    
    def Analyze_Opticalflow_Data(self):
        dfile  = glob.glob("{}/*.csv".format(self.dir_result))
        n_csv  = len(dfile)
        csvdata = np.genfromtxt(dfile[0], delimiter=",")
        
#        plt.figure('test')
#        plt.imshow(csvdata[0], origin='lower',aspect='auto')
        for i_file in range(n_csv):
            self.csv2vtk(filename = dfile[i_file])
        return(csvdata)
    
    def csv2vtk(self, filename, skiprows = 0):
        with open(filename,'r') as f:
            for i in range(skiprows):
                f.readline()
            x = []
            y = []
            value  = []
            flow_x = []
            flow_y = []
            for line0 in f:
                line = line0[0:len(line0)-1]
                item = line.split(",")
                x.append(item[0])
                y.append(item[1])
                value.append(item[2])
                flow_x.append(item[3])
                flow_y.append(item[4])
            f.close()
            # output
            basename = os.path.splitext(filename)[0]
            f = open(basename + ".vtk", "w")
            f.write("# vtk DataFile Version 2.0\n")
            f.write(basename + "\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write("POINTS %d float\n" % len(x))
            num = len(x)
            for i in range(0, num):
                f.write(x[i] + " " + y[i] + " 0\n")
            f.write("CELLS %d %d\n" % (num, 2*num))
            for i in range(0, num):
                f.write("1 %d\n" % i)
            f.write("CELL_TYPES %d\n" % num)
            for i in range(0, num):
                f.write("1\n")
            f.write("POINT_DATA %d\n" % num)
            f.write("SCALARS point_scalars float\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(0, num):
                f.write(value[i] + "\n")
            f.write("SCALARS flow_x float\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(0, num):
                f.write(flow_x[i] + "\n")
            f.write("SCALARS flow_y float\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(0, num):
                f.write(flow_y[i] + "\n")
        return()

    

if __name__ == "__main__":
    # =========================================================================
    # Input parameters
    # =========================================================================
    # Origami Dynamic Simulation based on Project Chrono -------------------
    # Input file
#    video_name = 'Movies/Chronos_demo.mp4'
    video_name = 'Movies/scale.vib.15hz.mp4'
#    video_name = 'Movies/wing.test.2.mp4'
    
    dir_result = 'Result_opencv'
    
    # Subdivision for opticalflow analysis
    n_width  = 60
    n_height = 30
    
    dic = DIC_Analysis(video_name, dir_result)
    
    # =========================================================================
    # DIC Analysis
    # =========================================================================
    # Opticalflow
    dic.DIC_OpticalFlow(n_width, n_height)
    
    csvdata = dic.Analyze_Opticalflow_Data()
    
    plt.show()
    
    
    
    
    
    
    
