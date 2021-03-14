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
 
# =======================
#   General parameters
# =======================


JTAG_PACKET_LEN = 48*10 #10ms
BUF_PACKET_LEN = 10  # 100ms

FRAME_LEN_WIDTH = np.round(np.log2(JTAG_PACKET_LEN*BUF_PACKET_LEN)).astype(int)

SHOW_FRAME_NUM = 10     #1s
BUF_MAX_LEN = 10
NUM_CHS = 40

ANGLE = 90.
CH_TESTED = 0x00

MODE = "frame" # "frame" or "packet"

# =======================
#    Local parameters
# =======================

MIC_IF_BASE = 0x4080000
SHOW_FRAME = JTAG_PACKET_LEN*BUF_PACKET_LEN*SHOW_FRAME_NUM

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

if MODE == "frame":
    # Microphone frame mode enable
    mic.set_frame_tag(10)

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
mic.set_bf_angle(np.pi*ANGLE/180., verbose=True)
mic.show_config()

# Start Bytestream Server
jtag_master.StartBytestreamServer()

# Queue
queue = Queue.Queue()

# Create Bytetream client
jtag_bytestream = jtag.BytestreamClient(queue, jtag_packet_len=JTAG_PACKET_LEN, buf_packet_len=BUF_PACKET_LEN, buf_maxlen=BUF_MAX_LEN, mode=MODE)

# Define client as daemon
jtag_bytestream.setDaemon(True)

# Flush FIFO
jtag_bytestream.flushFIFO(True)

# Mic enable
mic.enable(True)

# Start Bytestream client
jtag_bytestream.start()

# Mic config
mic.show_config()

# Data container
y_data = np.zeros(SHOW_FRAME)

# =======================
#      Plot data
# =======================

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

period = time.time()
    
def plot_data(i):

    global y_data
    global start
    global period

    data_int, _  = np.array(jtag_bytestream.getDataN(3))

    y_data = np.concatenate((y_data[-(SHOW_FRAME-len(data_int)):],data_int))
      
    ax1.clear()
    ax1.plot(y_data)
    ax1.grid()
    ax1.set_ylim(-2**15-1000, 2**15+1000)

    print 'period: %.2f ms' %((time.time()-period)*1e3) 
    period = time.time()

# =======================
#     Run animation
# =======================
an1 = animation.FuncAnimation(fig, plot_data, interval=250)
plt.show()
