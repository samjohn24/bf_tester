#! /usr/bin/env python2
"""
 +FHDR-------------------------------------------------------------------------
 FILE NAME      : mic_energy.py
 AUTHOR         : Sammy Carbajal
 ------------------------------------------------------------------------------
 PURPOSE
  Script to show audio output at real-time in Beaglebone Black board.
 -FHDR-------------------------------------------------------------------------
"""

import sys
sys.path.append('../ip/avalon_st_jtag/system-console/jtag_client/python/jtag_client')
sys.path.append('../ip/mic_if/hal/avalon_st_jtag/python')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import time
import Queue
import jtag_client as jtag
import mic_if_hal as mic_if
import mpl_toolkits.mplot3d.axes3d as axes3d
 
# =======================
#   General parameters
# =======================

FRAME_MS_LEN = 10
BUF_MAX_LEN = 10

NUM_PTS_POLAR = 9
NUM_PTS_AZIM = 5

VISIBLE_RAD = 0.5

MAX_PERC = 90

PLOT_POLAR = True
#PLOT_POLAR = False

# =======================
#    Local parameters
# =======================

MIC_IF_BASE = 0x4080000

FRAME_1MS_LEN = 48.
FRAME_LEN_WIDTH = np.round(np.log2(FRAME_1MS_LEN*FRAME_MS_LEN)).astype(int)
FRAME_DELAY = (2**FRAME_LEN_WIDTH)/FRAME_1MS_LEN*0.001

NUM_MIC_X = 8
NUM_MIC_Z = 5
NUM_CHS = NUM_MIC_X*NUM_MIC_Z

CH_TESTED = 0xff

MODE = "frame_buffer_tag" # "frame", "packet" or "frame_buffer_tag"

if NUM_PTS_POLAR < 3:
    ANGLES_POLAR = np.array([np.pi/2])
    NUM_PTS_POLAR = 1
else:
    ANGLES_POLAR = np.linspace(0, np.pi, NUM_PTS_POLAR)

if NUM_PTS_AZIM < 3:
    ANGLES_AZIM = np.array([0])
    NUM_PTS_AZIM = 1
else:
    ANGLES_AZIM = np.linspace(-np.pi/2, np.pi/2, NUM_PTS_AZIM)

# =======================
#      Configuration
# =======================
 
# Open JTAG Master client
jtag_master = jtag.AlteraJTAGClient()

# Mics failing
mic_failing = [6]

# Microphone interface
mic = mic_if.mic_if_hal(jtag_master,num_chs=NUM_CHS, MIC_IF_BASE=MIC_IF_BASE, mic_failing=mic_failing)

# Mic disable
mic.enable(False)

if MODE in ["frame","frame_buffer_tag"]:
    # Microphone frame mode enable
    mic.set_frame_tag(0)

    # Microphone frame mode enable
    mic.set_frame_len(FRAME_LEN_WIDTH)
    
    # Microphone frame mode enable
    mic.frame_mode(True)

# Microphone initialization
mic.init(clk="1.25M")

# Init array
mic.init_array_bbb_30()

# Select beamformer channel
mic.avalon_st_bytestream(True, CH_TESTED)
mic.set_bf_angle(np.pi/2., verbose=True)
mic.show_config()

# Start Bytestream Server
jtag_master.StartBytestreamServer()

# Queue
queue = Queue.Queue()

# Create Bytetream client
jtag_bytestream = jtag.BytestreamClient(queue, buf_maxlen=BUF_MAX_LEN, buf_num=NUM_PTS_POLAR*NUM_PTS_AZIM, mode=MODE)

# Define client as daemon
jtag_bytestream.setDaemon(True)

# Flush FIFO
jtag_bytestream.flushFIFO(True)

# Create mic_if_beamformer object
mic_if_beamformer = mic_if.mic_if_beamformer(mic, angles_polar = ANGLES_POLAR, angles_azim = ANGLES_AZIM, delay_sec = 0.05)

# Define mic_if_beamformer as deamon
mic_if_beamformer.setDaemon(True)

# Mic config
#mic.show_config()

# Mic enable
mic.enable(True)

# Start mic_if_beamformer
mic_if_beamformer.start()

# Start Bytestream client
jtag_bytestream.start()

# =======================
#      Plot data
# =======================

fig = plt.figure()

period = time.time()

energy = np.zeros((NUM_PTS_AZIM, NUM_PTS_POLAR))

max_val = 2**15
max_energy_val = max_val

theta, phi = ANGLES_POLAR, ANGLES_AZIM
THETA, PHI = np.meshgrid(theta, phi)

data_int = 0

if PLOT_POLAR:

    if NUM_PTS_POLAR != 1:
        if NUM_PTS_AZIM != 1:
            ax1 = fig.add_subplot(1,2,1, projection='polar')
            title = "Energy vs $\\theta$ angle\n ($\phi$ constant)"
            ax1.set(xlim=(0, np.pi), ylim=(0,VISIBLE_RAD), title=title)
            line1      = [ax1.plot(ANGLES_POLAR, energy[0,:], label="$\phi$=%.1f"%(round(ANGLES_AZIM[i]*180/np.pi)))[0] for i in range(NUM_PTS_AZIM)]
            ax1.legend(loc='upper left')
        else:
            ax1 = fig.add_subplot(1,1,1, projection='polar')
            title = "Energy vs $\\theta$ angle\n ($\phi$=0)"
            ax1.set(xlim=(0, np.pi), ylim=(0,VISIBLE_RAD), title=title)
            line1      = [ax1.plot(ANGLES_POLAR, energy[0,:])[0] for i in range(NUM_PTS_AZIM)]

    if NUM_PTS_AZIM != 1:
        if NUM_PTS_POLAR != 1:
            ax2 = fig.add_subplot(1,2,2, projection='polar')
            title="Energy vs $\phi$ angle\n ($\\theta$ constant)"
            ax2.set(xlim=(-np.pi/2, np.pi/2), ylim=(0,VISIBLE_RAD), title=title)
            line2      = [ax2.plot(ANGLES_AZIM, energy[:,0], label="$\\theta$=%.1f"%(round(ANGLES_POLAR[i]*180/np.pi)))[0] for i in range(NUM_PTS_POLAR)]
            ax2.legend(loc='lower right')
        else:
            ax2 = fig.add_subplot(1,1,1, projection='polar')
            title="Energy vs $\phi$ angle\n ($\\theta$=0)"
            ax2.set(xlim=(-np.pi/2, np.pi/2), ylim=(0,VISIBLE_RAD), title=title)
            line2      = [ax2.plot(ANGLES_AZIM, energy[:,0])[0] for i in range(NUM_PTS_POLAR)]

else:
    angles_polar_deg = ANGLES_POLAR*180./np.pi
    angles_azim_deg = ANGLES_AZIM*180./np.pi

    if NUM_PTS_POLAR != 1:
        if NUM_PTS_AZIM != 1:
            ax1 = fig.add_subplot(1,2,1)
            title = "Energy vs $\\theta$ angle\n ($\phi$ constant)"
        else:
            ax1 = fig.add_subplot(1,1,1)
            title = "Energy vs $\\theta$ angle\n ($\phi$=0)"

        ax1.set(xlim=(0, 180), ylim=(0,VISIBLE_RAD), title=title)

        ax1.grid(True)
        
        line1        = [ax1.plot(angles_polar_deg, energy[0,:])[0] for i in range(NUM_PTS_AZIM)]

    if NUM_PTS_AZIM != 1:
        if NUM_PTS_POLAR != 1:
            ax2 = fig.add_subplot(1,2,2)
            title="Energy vs $\phi$ angle\n ($\\theta$ constant)"
        else:
            ax2 = fig.add_subplot(1,1,1)
            title="Energy vs $\phi$ angle\n ($\\theta$=0)"

        ax2.set(xlim=(-90, 90), ylim=(0,VISIBLE_RAD), title=title)

        ax2.grid(True)

        line2        = [ax2.plot(angles_azim_deg, energy[:,0])[0] for i in range(NUM_PTS_POLAR)]


line1_max    = ax1.plot([0], [0], 'o', color="y", label="Max. points")[0]
line1_center = ax1.plot([0], [0], '-x', color='g')[0]
line1_center_text = ax1.text(0, 0 , '0', color='g')
#ax1.legend(loc='upper left')

line2_max    = ax2.plot([0], [0], 'o', color="y", label="Max. points")[0]
line2_center = ax2.plot([0], [0], '-x', color='g')[0]
line2_center_text = ax2.text(0, 0 , '0', color='g')
#ax2.legend(loc='lower right')

def plot_data(i):

    global y_data
    global start
    global period

    for j in range(NUM_PTS_AZIM):
        for i in range(NUM_PTS_POLAR):
            try:
                data_int, _  = jtag_bytestream.getDataN(1, frame_tag=j*NUM_PTS_POLAR+i)
                energy[j,i] = np.std(data_int)/max_energy_val
            except IndexError:
                continue

    # Upper 25% 
    X = energy * np.cos(PHI) * np.cos(THETA)
    Y = energy * np.cos(PHI) * np.sin(THETA)
    Z = energy * np.sin(PHI)

    R = np.sqrt(X**2+Y**2+Z**2)

    R_perc = R >= np.percentile(R, MAX_PERC)

    # Center
    center_x, center_y, center_z = np.mean(X[R_perc]), np.mean(Y[R_perc]), np.mean(Z[R_perc])

    center_theta = np.arctan(center_y/center_x)
    if center_x < 0:
        center_theta += np.pi
    center_phi = np.arctan(center_z/np.sqrt(center_x**2+center_y**2))

    # Plotting
    if NUM_PTS_POLAR != 1:
        for j in range(NUM_PTS_AZIM):
            line1[j].set_ydata(energy[j,:])

    if NUM_PTS_AZIM != 1:
        for i in range(NUM_PTS_POLAR):
            line2[i].set_ydata(energy[:,i])

    if PLOT_POLAR:
        line1_max.set_data(THETA[R_perc], R[R_perc])
        line1_center.set_data([center_theta]*2, [0,VISIBLE_RAD])
        line1_center_text.set_position((center_theta, VISIBLE_RAD))

        line2_max.set_data(PHI[R_perc], R[R_perc])
        line2_center.set_data([center_phi]*2, [0,VISIBLE_RAD])
        line2_center_text.set_position((center_phi, VISIBLE_RAD))
    else:
        line1_max.set_data(THETA[R_perc]*180/np.pi, R[R_perc])
        line1_center.set_data([center_theta*180/np.pi]*2, [0,VISIBLE_RAD])
        line1_center_text.set_position((center_theta*180/np.pi, VISIBLE_RAD))

        line2_max.set_data(PHI[R_perc]*180/np.pi, R[R_perc])
        line2_center.set_data([center_phi*180/np.pi]*2, [0,VISIBLE_RAD])
        line2_center_text.set_position((center_phi*180/np.pi, VISIBLE_RAD))

    line1_center_text.set_text("(%.1f)"%(center_theta*180/np.pi))
    line2_center_text.set_text("(%.1f)"%(center_phi*180/np.pi))

    print 'period: %.2f ms' %((time.time()-period)*1e3) 
    period = time.time()

# =======================
#     Run animation
# =======================
an1 = animation.FuncAnimation(fig, plot_data, interval=100)
plt.show()
