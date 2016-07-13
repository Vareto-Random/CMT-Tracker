import cv2

import VARtracker
import numpy as np

def main():
    VARmethod('../frames/', 247, [405, 160], [450, 275])


def VARmethod(folderPath, finalFrame, topLeft, bottomRight):
    CMT1 = VARtracker.CMT()

    CMT1.estimate_scale = True
    CMT1.estimate_rotation = True

    numericalList = [i for i in range(0, finalFrame + 1)]
    stringList = []
    for number in numericalList:
        stringList.append(str(number) + '.jpeg')

    pause_time = 1

    print 'using', topLeft, bottomRight, 'as init bb'

    framePath = folderPath + '/' + stringList[0]
    im0 = cv2.imread(framePath)
    im_gray0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    VARtracker.initialise(CMT1, im_gray0, topLeft, bottomRight)

    frame = 1
    while True:
        print frame

        # Read image
        framePath = folderPath + '/' + stringList[frame]
        im = cv2.imread(framePath)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        res1 = VARtracker.process_frame(CMT1, im_gray)

        # Display results
        im_draw = np.copy(im)
        if res1.has_result:
            cv2.line(im_draw, res1.tl, res1.tr, (255, 0, 0), 4)
            cv2.line(im_draw, res1.tr, res1.br, (255, 0, 0), 4)
            cv2.line(im_draw, res1.br, res1.bl, (255, 0, 0), 4)
            cv2.line(im_draw, res1.bl, res1.tl, (255, 0, 0), 4)

        cv2.imshow('main', im_draw)
        cv2.waitKey(pause_time)

        # Remember image
        im_prev = im_gray
        frame += 1

if __name__ == "__main__":
    main()