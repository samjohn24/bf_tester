#! /usr/bin/env python
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
sys.path.append('../../../hardware/ip/avalon_st_jtag/system-console/jtag_client/python/jtag_client')
sys.path.append('../../../hardware/ip/mic_if/hal/avalon_st_jtag/python')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib import cm, colors
import time
import Queue
import jtag_client as jtag
import mic_if_hal as mic_if
import mpl_toolkits.mplot3d.axes3d as axes3d
 
# =======================
#   General parameters
# =======================

FRAME_MS_LEN = 5
BUF_MAX_LEN = 10

NUM_PTS_POLAR = 9
NUM_PTS_AZIM = 5

MAX_PERC = 90
VISIBLE_RAD = 0.5

facecolors = False

ANGLES_POLAR = np.linspace(0, np.pi, NUM_PTS_POLAR)

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

MODE = "frame_buffer_tag"

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
mic_if_beamformer = mic_if.mic_if_beamformer(mic, angles_polar = ANGLES_POLAR, angles_azim = ANGLES_AZIM, delay_sec = FRAME_DELAY)

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

if mic.bf_pdm:
    max_energy_val = 2**4
else:
    max_energy_val = 2**15

angles_polar_deg = ANGLES_POLAR*180./np.pi
angles_azim_deg = ANGLES_AZIM*180./np.pi

data_int = 0

# Plot parameters

theta, phi = ANGLES_POLAR, ANGLES_AZIM
THETA, PHI = np.meshgrid(theta, phi)

ax = fig.add_subplot(1,1,1, projection='3d')

axis_opt = {'xlim':(-VISIBLE_RAD, VISIBLE_RAD), \
            'ylim':(0, VISIBLE_RAD), \
            'zlim':(-VISIBLE_RAD, VISIBLE_RAD), \
            'xlabel':'X', 'ylabel':'Y', \
            'title':'Real-time DoA detector\n(DAS Beamformer)'}

ax.set(**axis_opt)
ax.set_axis_off()

plot_args = { 'rstride':1, 'cstride':1, 'linewidth':1.0, 'antialiased':True, 'alpha':1.0, 'edgecolor':'k', 'shade':False}

# Plot sphere parameters
SPH_PTS = 20

theta_sph, phi_sph = np.linspace(0, np.pi, SPH_PTS), np.linspace(-np.pi/2, np.pi/2, SPH_PTS) 
THETA_SPH, PHI_SPH = np.meshgrid(theta_sph, phi_sph)

X_SPH, Y_SPH, Z_SPH = VISIBLE_RAD*np.cos(PHI_SPH)*np.cos(THETA_SPH), \
                      VISIBLE_RAD*np.cos(PHI_SPH)*np.sin(THETA_SPH), \
                      VISIBLE_RAD*np.sin(PHI_SPH)

plot_sph_args = { 'rstride':1, 'cstride':1, 'linewidth':0.1, 'linestyle':'--', 'color':'black'}


# Plot board plane
PLANE_PTS = 2

x_max = 2*VISIBLE_RAD*0.1*NUM_MIC_X
z_max = 2*VISIBLE_RAD*0.1*NUM_MIC_Z

x_plane, z_plane = np.linspace(-x_max*0.5, x_max*0.5,PLANE_PTS), np.linspace(-z_max*0.5, z_max*0.5, PLANE_PTS) 
X_PLANE, Z_PLANE = np.meshgrid(x_plane, z_plane)
Y_PLANE = np.zeros(X_PLANE.shape)

plot_plane_args = { 'rstride':1, 'cstride':1, 'linewidth':0.5, 'linestyle':'--', 'color':'green', 'alpha':0.5}

# Plot ref X=0 plane and 
RPLANE_PTS = 10
x_plane, y_plane, z_plane = np.linspace(-VISIBLE_RAD, VISIBLE_RAD, RPLANE_PTS), \
                            np.linspace(0, VISIBLE_RAD, RPLANE_PTS), \
                            np.linspace(-VISIBLE_RAD, VISIBLE_RAD, RPLANE_PTS) 

Z_RPLANE0, Y_RPLANE0 = np.meshgrid(z_plane, y_plane)
X_RPLANE0 = np.zeros(Y_RPLANE0.shape)

X_RPLANE1, Y_RPLANE1 = np.meshgrid(x_plane, y_plane)
Z_RPLANE1 = np.zeros(Y_RPLANE1.shape)

plot_rplane_args = { 'rstride':1, 'cstride':1, 'linewidth':0.2, 'linestyle':'--', 'color':'red'}

def plot_data(i):

    global period

    for j in range(NUM_PTS_AZIM):
        for i in range(NUM_PTS_POLAR):
            try:
                data_int, _  = jtag_bytestream.getDataN(1, frame_tag=j*NUM_PTS_POLAR+i)
                energy[j,i] = np.std(data_int)/max_energy_val
            except IndexError:
                continue

    # Surface Plotting
    X = energy * np.cos(PHI) * np.cos(THETA)
    Y = energy * np.cos(PHI) * np.sin(THETA)
    Z = energy * np.sin(PHI)

    R = np.sqrt(X**2+Y**2+Z**2)

    # Upper 25% 
    R_perc = R >= np.percentile(R, MAX_PERC)

    X_PERC_PROJ = VISIBLE_RAD * np.cos(PHI[R_perc]) * np.cos(THETA[R_perc])
    Y_PERC_PROJ = VISIBLE_RAD * np.cos(PHI[R_perc]) * np.sin(THETA[R_perc])
    Z_PERC_PROJ = VISIBLE_RAD * np.sin(PHI[R_perc])

    # Center

    center_x, center_y, center_z = np.mean(X[R_perc]), np.mean(Y[R_perc]), np.mean(Z[R_perc])

    center_theta = np.arctan(center_y/center_x)
    if center_x < 0:
        center_theta += np.pi
    center_phi = np.arctan(center_z/np.sqrt(center_x**2+center_y**2))

    X_CENTER = VISIBLE_RAD * np.cos(center_phi) * np.cos(center_theta)
    Y_CENTER = VISIBLE_RAD * np.cos(center_phi) * np.sin(center_theta)
    Z_CENTER = VISIBLE_RAD * np.sin(center_phi)

    STR_CENTER = "(%.1f,%.1f)" %(center_theta*180/np.pi, center_phi*180/np.pi)

    # Plotting
    ax.clear()
            
    ax.set(**axis_opt)
    ax.set_axis_off()

    if facecolors:
        f_colors = R/np.max(R)
        ax.plot_surface( X, Y, Z, facecolors=cm.seismic(f_colors), **plot_args)
    else:
        ax.plot_surface( X, Y, Z, color='b', **plot_args)

    ax.plot_wireframe(X_SPH, Y_SPH, Z_SPH, **plot_sph_args)
    ax.plot_surface(X_PLANE, Y_PLANE, Z_PLANE, **plot_plane_args)
    ax.plot_wireframe(X_RPLANE0, Y_RPLANE0, Z_RPLANE0, **plot_rplane_args)
    ax.plot_wireframe(X_RPLANE1, Y_RPLANE1, Z_RPLANE1, **plot_rplane_args)

    ax.scatter( X[R_perc], Y[R_perc], Z[R_perc], marker='.', s=30, c='yellow', alpha=1.0, label='Max. energy points')
    ax.scatter( X_PERC_PROJ, Y_PERC_PROJ, Z_PERC_PROJ, marker='x', s=30, c='blue', alpha=1.0, label='Max. energy points projection')

    ax.plot( [0, X_CENTER.flatten()], [0, Y_CENTER.flatten()], [0, Z_CENTER.flatten()], marker='x', c='red', alpha=0.5, label='Max. energy mean projection')
    ax.text( X_CENTER, Y_CENTER, Z_CENTER, STR_CENTER, color='red', alpha=1.0, )

    ax.legend(loc='upper right')

    print 'period: %.2f ms' %((time.time()-period)*1e3) 
    period = time.time()

# =======================
#     Run animation
# =======================
an1 = animation.FuncAnimation(fig, plot_data, interval=1)
plt.show()
