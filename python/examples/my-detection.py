#!/usr/bin/python3

import datetime as dt
import time

import jetson.inference
import jetson.utils


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.6)
camera = jetson.utils.videoSource("/dev/video0") 
display = jetson.utils.videoOutput("display://0")

start = dt.datetime.now()
checkpoint = start
i = 0
while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	fps = int(net.GetNetworkFPS())
	display.SetStatus("Object Detection | {} FPS".format(fps))
	if i % 100 == 0: 
		if  (dt.datetime.now() - checkpoint).total_seconds() > 30:
			ds = (dt.datetime.now() - start).total_seconds()
			h, rem = divmod(ds, 3600)
			m, s = divmod(rem, 60)
			if h > 0:
				msg = '{} hours {} minutes {} seconds'.format(int(h), int(m), int(s))
			elif m > 0:
				msg = '{} minutes {} seconds'.format(int(m), int(s))
			else:
				msg = '{} seconds'.format(int(s))
			print(f'- running for: {msg}, Current FPS: {fps}')

			checkpoint = dt.datetime.now()
			i = 0
	i += 1

