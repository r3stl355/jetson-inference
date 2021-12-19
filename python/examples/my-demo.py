#!/usr/bin/python3

import datetime as dt
import argparse

import jetson.inference
import jetson.utils

from segnet_utils import *

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2")
parser.add_argument("--threshold", type=float, default=0.6)
parser.add_argument("--segment", action="store_true")
parser.add_argument("--segment-network", type=str, default="fcn-resnet18-cityscapes")
parser.add_argument("--segment-filter-mode", dest="filer_mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--segment-visualize", dest="visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--segment-ignore-class", dest="ignore_class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--segment-alpha", dest="alpha", type=float, default=100.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--segment-stats", dest="stats", action="store_true", help="compute statistics about segmentation mask class output")

opt = parser.parse_args()

def run_detect(img):
	net.Detect(img)
	return img

def run_segment(img):
	# allocate buffers for this size image
	buffers.Alloc(img.shape, img.format)

	# process the segmentation network
	net.Process(img, ignore_class=opt.ignore_class)

	# generate the overlay
	if buffers.overlay:
		net.Overlay(buffers.overlay, filter_mode=opt.filter_mode)

	# generate the mask
	if buffers.mask:
		net.Mask(buffers.mask, filter_mode=opt.filter_mode)

	# composite the images
	if buffers.composite:
		jetson.utils.cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
		jetson.utils.cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

	return buffers.output

net, buffers = None, None

if opt.segment:
	# Runnig segmentation
	opt.network = opt.segment_network
	net = jetson.inference.segNet(opt.network)
	net.SetOverlayAlpha(opt.alpha)

	# Segmentation uses a buffer manager
	buffers = segmentationBuffers(net, opt)
else:
	net = jetson.inference.detectNet(opt.network, threshold=opt.threshold)

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
	output.SetStatus("{} | {} FPS".format(opt.network ,fps))

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

