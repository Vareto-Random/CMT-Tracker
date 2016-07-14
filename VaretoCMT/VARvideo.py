import argparse
import cv2 as cv
import numpy as np
from multiprocessing import Pool
from threading import Thread
import os
import sys
import time

import VARtracker

import util

import Queue

CMT1 = VARtracker.CMT()
CMT2 = VARtracker.CMT()

OUTPUT_FILE = "out_boxes.txt"
open(OUTPUT_FILE, 'w').close()

parser = argparse.ArgumentParser(description='Track an object.')

parser.add_argument('inputpath', nargs='?', help='The input path.')
parser.add_argument('--no-preview', dest='preview', action='store_const', const=False, default=None,
                    help='Disable preview')
parser.add_argument('--no-scale', dest='estimate_scale', action='store_false', help='Disable scale estimation')
parser.add_argument('--with-rotation', dest='estimate_rotation', action='store_true', help='Enable rotation estimation')
parser.add_argument('--bbox', dest='bbox', help='Specify initial bounding box.')
parser.add_argument('--pause', dest='pause', action='store_true', help='Pause after every frame and wait for any key.')
parser.add_argument('--output-dir', dest='output', help='Specify a directory for output data.')
parser.add_argument('--quiet', dest='quiet', action='store_true',
                    help='Do not show graphical output (Useful in combination with --output-dir ).')
parser.add_argument('--skip', dest='skip', action='store', default=None, help='Skip the first n frames', type=int)

args = parser.parse_args()

CMT1.estimate_scale = args.estimate_scale
CMT2.estimate_scale = args.estimate_scale
CMT1.estimate_rotation = args.estimate_rotation
CMT2.estimate_rotation = args.estimate_rotation

if args.pause:
    pause_time = 0
else:
    pause_time = 10

if args.output is not None:
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    elif not os.path.isdir(args.output):
        raise Exception(args.output + ' exists, but is not a directory')

# Clean up
cv.destroyAllWindows()

if args.inputpath is not None:
    # If a path to a file was given, assume it is a single video file
    if os.path.isfile(args.inputpath):
        cap = cv.VideoCapture(args.inputpath)
        # Skip first frames if required
        if args.skip is not None:
            cap.set(cv.CV_CAP_PROP_POS_FRAMES, args.skip)

    # Otherwise assume it is a format string for reading images
    else:
        cap = util.FileVideoCapture(args.inputpath)
        # Skip first frames if required
        if args.skip is not None:
            cap.frame = 1 + args.skip

    # Check if videocapture is working
    if not cap.isOpened():
        print 'Unable to open video input.'
        sys.exit(1)

    # Read first frame
    status, im0 = cap.read()
    im_gray0 = cv.cvtColor(im0, cv.COLOR_BGR2GRAY)
    im_draw = np.copy(im0)

# if args.bbox is not None:
#     # Try to disassemble user specified bounding box
#     values = args.bbox.split(',')
#     try:
#         values = [int(v) for v in values]
#     except:
#         raise Exception('Unable to parse bounding box')
#     if len(values) != 4:
#         raise Exception('Bounding box must have exactly 4 elements')
#     bbox = np.array(values)
#
#     # Convert to point representation, adding singleton dimension
#     bbox = util.bb2pts(bbox[None, :])
#
#     # Squeeze
#     bbox = bbox[0, :]
#
#     tl = bbox[:2]
#     br = bbox[2:4]

tl1 = [405, 160]
br1 = [450, 275]
tl2 = [255, 100]
br2 = [275, 155]

print 'using', tl1, br1, 'as init bb'

VARtracker.initialise(CMT1, im_gray0, tl1, br1)
# VARtracker.initialise(CMT2, im_gray0, tl2, br2)

frame = 1
while True:
    pool = Pool(processes=4)
    print frame

    # Read image
    status, im = cap.read()
    if not status:
        break
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    im_draw = np.copy(im)

    tic = time.time()
    res1 = VARtracker.process_frame(CMT1, im_gray)
    # res2 = VARtracker.process_frame(CMT2, im_gray)

    # res1 = pool.apply_async(VARtracker.process_frame, (CMT2, im_gray))
    # res2 = pool.apply_async(VARtracker.process_frame, (CMT2, im_gray))
    # pool.close()
    # pool.join()
    # res1 = res1.get()
    # res2 = res2.get()
    toc = time.time()

    # Display results
    if res1.has_result:
        cv.line(im_draw, res1.tl, res1.tr, (255, 0, 0), 4)
        cv.line(im_draw, res1.tr, res1.br, (255, 0, 0), 4)
        cv.line(im_draw, res1.br, res1.bl, (255, 0, 0), 4)
        cv.line(im_draw, res1.bl, res1.tl, (255, 0, 0), 4)
    # if res2.has_result:
    #     cv.line(im_draw, CMT2.tl, CMT2.tr, (255, 0, 0), 4)
    #     cv.line(im_draw, CMT2.tr, CMT2.br, (255, 0, 0), 4)
    #     cv.line(im_draw, CMT2.br, CMT2.bl, (255, 0, 0), 4)
    #     cv.line(im_draw, CMT2.bl, CMT2.tl, (255, 0, 0), 4)

    if not args.quiet:
        cv.imshow('main', im_draw)
        cv.waitKey(pause_time)


    # Remember image
    im_prev = im_gray
    frame += 1