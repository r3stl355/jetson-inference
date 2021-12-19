#!/usr/bin/python3

import datetime as dt
import time

import jetson.inference
import jetson.utils

import argparse
import sys

from segnet_utils import *

DETECT_NET = "ssd-mobilenet-v2"
SEGMENT_NET = "fcn-resnet18-cityscapes"

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("--segment", type=bool,  action="store_true")

opt = parser.parse_args()

def run_detect(img):
	net.Detect(img)
	return img

def run_segment(img):
	# allocate buffers for this size image
	buffers.Alloc(img.shape, img.format)

	# process the segmentation network
	net.Process(img, ignore_class="void")

	# generate the overlay
	if buffers.overlay:
		net.Overlay(buffers.overlay, filter_mode="linear")

	# generate the mask
	if buffers.mask:
		net.Mask(buffers.mask, filter_mode="linear")

	# composite the images
	if buffers.composite:
		jetson.utils.cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
		jetson.utils.cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

	return buffers.output

net, buffers = None, None
title = f"Object Detection ({DETECT_NET})"
if opt.segment:
	# Runnig segmentation
	title = f"Segmentation ({SEGMENT_NET})"
	net = jetson.inference.segNet(SEGMENT_NET, sys.argv)
	net.SetOverlayAlpha(opt.alpha)

	# create buffer manager
	buffers = segmentationBuffers(net, opt)
else:
	net = jetson.inference.detectNet(DETECT_NET, threshold=0.6)

input = jetson.utils.videoSource("/dev/video0") 
output = jetson.utils.videoOutput("display://0")

start = dt.datetime.now()
checkpoint = start
i = 0
while output.IsStreaming():
	img = input.Capture()

	if opt.segment:
		res = run_segment(img)
	else:
		res = run_detect(img)
	
	output.Render(res)
	fps = int(net.GetNetworkFPS())
	output.SetStatus("{} | {} FPS".format(title ,fps))

	if i % 100 == 0: 
		if  (dt.datetime.now() - checkpoint).total_seconds() > 30:
			ds = (dt.datetime.now() - start).total_seconds()
			h, rem = divmod(ds, 3600)
			m, s = divmod(rem, 60)
			if h > 0:
				msg = '{} hours {} minutes'.format(int(h), int(m))
			elif m > 0:
				msg = '{} minutes'.format(int(m))
			else:
				msg = '{} seconds'.format(int(s))
			print(f'- running for: {msg}, Current FPS: {fps}')

			checkpoint = dt.datetime.now()
			i = 0
	i += 1

