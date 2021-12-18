#!/usr/bin/python3

import datetime as dt
import time

import jetson.inference
import jetson.utils


net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.6)
camera = jetson.utils.videoSource("/dev/video0") 
display = jetson.utils.videoOutput("display://0")

checkpoint = dt.datetime.now()
i = 0
while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	fps = net.GetNetworkFPS()
	display.SetStatus("Object Detection | {:.0f} FPS".format(fps))
	if i % 100 == 0:
		delta = dt.datetime.now() - checkpoint
		if  delta.total_seconds() > 30:
			h, rem = divmod(delta.total_seconds(), 3600)
			m, s = divmod(rem, 60)
			if h > 0:
				msg = '{:02} hours {:02} minutes {:02} seconds'.format(int(h), int(m), int(s))
			elif m > 0:
				msg = '{:02} minutes {:02} seconds'.format(int(m), int(s))
			else:
				msg = '{:02} seconds'.format(int(s))
			print(f'- running for: {msg}, Current FPS: {fps})')

			checkpoint = dt.datetime.now()
			i = 0
	i += 1

