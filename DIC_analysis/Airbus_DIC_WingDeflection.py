import numpy as np
import matplotlib.pyplot as plt
import cv2
#from scipy import optimize
#from scipy.integrate import ode
#from scipy import integrate
#from matplotlib import animation
#from ctypes import (CDLL, POINTER, ARRAY, c_void_p,
#                    c_int, byref,c_double, c_char,
#                    c_char_p, create_string_buffer)
#from numpy.ctypeslib import ndpointer
#import os
#import shutil   #High level file operation
#import glob     #Get file list by using wildcard
#
#from scipy import fftpack
#from scipy import signal
#import cmath
import time

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
plt.rcParams['legend.fontsize'] = 21        # Original fornt size is 8, multipy by above number
plt.rcParams['xtick.labelsize'] = 24        # Original fornt size is 8, multipy by above number
plt.rcParams['ytick.labelsize'] = 24        # Original fornt size is 8, multipy by above number
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams['figure.subplot.left']  = 0.2
#plt.rcParams['figure.subplot.right']  = 0.7
plt.rcParams['figure.subplot.bottom']  = 0.2
plt.rcParams['savefig.dpi']  = 300


class DIC_WingDeflection:
    def __init__(self,video_name, Play_speed, scale_image, track_type):
        self.video_name = video_name
        self.Play_speed = Play_speed
        self.scale_image = scale_image
        self.track_type  = track_type
        self.windowname = 'frame'
        # Parameters for the filter
        self.hsvLower = (0,70,70)   # For Red markers
        self.hsvUpper = (30,255,255) # For Red markers
        
        
    def Play_movie(self):
        self.cap = cv2.VideoCapture(self.video_name)
        self.fps = self.cap.get(5)
        self.interval  = round(self.cap.get(5))
        self.interval0 = round(self.cap.get(5))
        # Play movie-----------------------------------------------------------
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 1.0
        text1 = str(round(self.fps)) + 'fps'
        text3 = 's=stop,y=save, r=restart, b=back'
        while True:
            # Import one frame ------------------------------------------------
            ret, self.frame = self.cap.read()
            if( self.frame is None ): # if you reach to the end of the movie, stop the anlysis
                break
            
            # Resize the image for display ------------------------------------
            height = self.frame.shape[0]
            width  = self.frame.shape[1]
            self.frame = cv2.resize(self.frame,(int(self.scale_image*width), int(self.scale_image*height)))
            
            # Display image ---------------------------------------------------
            x_text = 700
            cv2.putText(self.frame,text1,(x_text,20),font, font_size,(0,0,0))
            n_frame = self.cap.get(1)
            text21 = 'Current frame # = {0:1d}'.format(int(n_frame))
            text22 = 'Time            = {0:1.2f} [s]'.format(float(n_frame)/self.fps)
            cv2.putText(self.frame,text21,(x_text,40),font, font_size,(0,0,0))
            cv2.putText(self.frame,text22,(x_text,60),font, font_size,(0,0,0))
            cv2.putText(self.frame,text3,(x_text,60),font, font_size,(0,0,0))
            cv2.imshow(self.windowname,self.frame)
            
            # Interval
            key = cv2.waitKey(self.interval)
            
            # End the process if "esc" is pressed
            if key == 0x1b: #"esc" key
                break
            elif key == 0x62: #"b" key
                self.interval = 0
                self.cap.set(1, n_frame-2)
            elif key == 0x73: #"s" key
                self.interval = 0
            elif key == 0x72: #"r" key
                self.interval = self.interval0
        
        self.cap.release()
        cv2.destroyAllWindows()
        return()
    
    def Track_WingDeflection(self, f_ROI, bbox, f_VideoSave, name_VideoSave, f_display = True):
        self.cap = cv2.VideoCapture(self.video_name)
        self.fps = self.cap.get(5)
        self.interval  = round(self.cap.get(5))
        self.interval0 = round(self.cap.get(5))
        # Play movie-----------------------------------------------------------
        font = cv2.FONT_HERSHEY_PLAIN
        font_size = 1.0
        text1 = str(round(self.fps)) + 'fps'
        text3 = 's=stop,y=save,r=restart,f=100frame foward,b=back'
        # Import the first frame
        ret, self.frame = self.cap.read()
        height = int(self.scale_image*self.frame.shape[0])
        width  = int(self.scale_image*self.frame.shape[1])
        print('width = {0}, height = {1}'.format(width, height) )
        
        # Set up for tracking -------------------------------------------------
        # Select tracking type
        if self.track_type=='KCF': # KCF (Kernelized Correlation Filters) Try this first!
            tracker = cv2.TrackerKCF_create()
        elif self.track_type=='MIL':
            tracker = cv2.TrackerMIL_create()
        elif self.track_type=='Boosting':
            tracker = cv2.TrackerBoosting_create()
        elif self.track_type=='TLD': # TLD (Tracking, Learning, and Detection)
            tracker = cv2.TrackerTLD_create()
        elif self.track_type=='MedianFlow':
            tracker = cv2.TrackerMedianFlow_create()
        else:
            print('Please select the Tracking type!')
            return()
        
        # Select the region of interest for tracking 
        while True:
            ret, self.frame = self.cap.read()
            if not ret:
                continue
            self.frame = cv2.resize(self.frame, (width, height) )
            if f_ROI==True:
                bbox = cv2.selectROI(self.frame, False)
                print('Selected bbox region = ({0}, {1}, {2}, {3})'.format(bbox[0], bbox[1], bbox[2], bbox[3]))
            ok = tracker.init(self.frame, bbox)
            cv2.destroyAllWindows()
            break
        
        # Set up for saving the movie------------------------------------------
        if f_VideoSave==True:
            fourcc   = cv2.VideoWriter_fourcc(*"MP4V")
            self.out = cv2.VideoWriter(name_VideoSave+'.mp4', fourcc, round(self.fps), (width, height) )
        
# =============================================================================
#         Start Tacking 
# =============================================================================
        time_DIC = []
        pos_cnt = []
        t_start = time.time()
        while True:
            # Import one frame ------------------------------------------------
            ret, self.frame = self.cap.read()
            n_frame = self.cap.get(1) 
            if( self.frame is None ): # if you reach to the end of the movie, stop the anlysis
                break
            
            # Resize the image for display ------------------------------------                           
            self.frame = cv2.resize(self.frame, (width, height) )
            
            
#            # Apply filter --------
#            hsv  = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
#            mask = cv2.inRange(hsv, self.hsvLower, self.hsvUpper)
#            mask = cv2.medianBlur(mask,7)
#            mask = cv2.bitwise_not(mask)
#            # Create Gray scale image
#            mask0    = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Tracking --------------------------------------------------------
            # update tracker
            track, bbox = tracker.update(self.frame)
            # Draw a box of the trackged object
            if track:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                # Calculate the center point
                x_cnt = bbox[0] + 0.5*bbox[2]
                y_cnt = bbox[1] + 0.5*bbox[3]
                pos_cnt.append([x_cnt, y_cnt])
                time_DIC.append(float(n_frame)/self.fps)
                # Draw the box and the center point
                cv2.rectangle(self.frame, p1, p2, (0,255,0), 2, 1)
                cv2.circle(self.frame, (int(x_cnt), int(y_cnt)), 4, (0,255,0), -1, 8, 0)
                cv2.putText(self.frame, "Tracking ({0})".format(self.track_type), (int(bbox[0]),int(bbox[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            else :
                # If tacking fails, give warning
                cv2.putText(self.frame, "Failure", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
                print('Tracking fails!!!')
            # Display image ---------------------------------------------------
            x_text = 500
            cv2.putText(self.frame,text1,(x_text,20),font, font_size,(0,0,0))
            text21 = 'Frame = {0:1d}'.format(int(n_frame))
            text22 = 'Time  = {0:1.2f} [s]'.format(float(n_frame)/self.fps)
            cv2.putText(self.frame,text21,(x_text,40),font, font_size,(0,0,0))
            cv2.putText(self.frame,text22,(x_text,60),font, font_size,(0,0,0))
            cv2.putText(self.frame,text3,(x_text,80),font, font_size,(0,0,0))
            # Plot the image
            if f_display==True:
                cv2.imshow(self.windowname, self.frame)            
#            cv2.imshow("Masked image", mask0)
            # Save the image
            if f_VideoSave==True:
                self.out.write(self.frame)
            
            # Interval
            key = cv2.waitKey(self.interval)
            self.cap.set(1, n_frame+self.Play_speed )
            # End the process if "esc" is pressed
            if key == 0x1b: #"esc" key
                break
            elif key == 0x62: #"b" key
                self.interval = 0
                self.cap.set(1, n_frame-2)
            elif key == 0x66: #"f" key
                self.interval = 0
                self.cap.set(1, n_frame+100)
            elif key == 0x73: #"s" key
                self.interval = 0
            elif key == 0x72: #"r" key
                self.interval = self.interval0

        self.cap.release()
        if f_VideoSave==True:
            self.out.release()
        cv2.destroyAllWindows()
        
        # Calculate the elapsed time
        elapsed_time = time.time() - t_start
        print ("elapsed_time:{0:1.2f}".format(elapsed_time) + "[sec]")
        
        # Save the data
        time_DIC = np.array(time_DIC)
        pos_cnt  = np.array(pos_cnt) 
        with open('{0}.txt'.format(name_VideoSave), 'w') as f:
            f.write('Time [sec], Vertical displacement [pixels], Horizontal displacement [pixcels] \n')
            for i in range(len(time_DIC)):
                f.write("{0}, {1}, {2}\n".format(time_DIC[i], pos_cnt[i,0], pos_cnt[i,1]) )
        return(time_DIC, pos_cnt )
    
    def Analysis_DIC_Results(self, filename):
        # Import the data
        data_txt = np.genfromtxt(filename, delimiter=",", skip_header=1)
        time_DIC = data_txt[:,0]
        x_disp   = data_txt[:,1]
        y_disp   = data_txt[:,2]
        
        # Analyze the data
        disp_TTL = np.sqrt( (x_disp-np.mean(x_disp[0:10]))**2 + (y_disp-np.mean(y_disp[0:10]))**2 )
        
        # Plot
        plt.figure('Displacement', figsize=(8, 10) )
        plt.subplot(211)
        plt.plot(time_DIC, x_disp, 'b')
        plt.ylabel('Horizontal disp. [pixel]')
        plt.xlim(xmin=0)
        plt.subplot(212)
        plt.plot(time_DIC, y_disp, 'b')
        plt.xlabel('Time [s]')
        plt.ylabel('Vertical disp. [pixel]')
        plt.xlim(xmin=0)
        
        plt.figure('Tip deflection')
        plt.plot(time_DIC, -(y_disp-np.mean(y_disp[0:10])), 'b-')
        plt.xlabel('Time [s]')
        plt.ylabel('Tip deflection [pixel]')
        plt.xlim(xmin=0)
        
        plt.figure('Total displacement')
        plt.plot(time_DIC, disp_TTL, 'b')
        plt.xlabel('Time [s]')
        plt.ylabel('Total disp. [pixel]')
        plt.xlim(xmin=0)
        return()

if __name__ == '__main__':
    # =============================================================================
    # Input parameters
    # =============================================================================
    video_name  = 'MovieFiles/B777_InflightWingFlex_rev.mp4' # 'MovieFiles/Airbus_A320_WingView_GreatFlyer.mp4' 'MovieFiles/B777_InflightWingFlex_rev.mp4'
    Play_speed  = 0    # 0=Normal speed
    scale_image = 0.8
    track_type  = 'KCF'
    # Create the class
    dic = DIC_WingDeflection(video_name, Play_speed, scale_image, track_type)
    
    # =============================================================================
    # Start analysis
    # =============================================================================
    name_datasave = 'AnalyzedData_test'
    # Check the movie file
#    dic.Play_movie()
    
    # DIC analysis to extract the height change
    time_DIC, pos_cnt = dic.Track_WingDeflection(
            f_ROI = False, bbox = (161, 31, 129, 48), #(256, 61, 62, 82) for airbus, (161, 31, 129, 48) for boeing
            f_VideoSave=True, name_VideoSave=name_datasave, f_display = True)
    
#    # Plot the extracted data
    dic.Analysis_DIC_Results(filename=name_datasave+'.txt')

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    