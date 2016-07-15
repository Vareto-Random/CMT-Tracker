import cv2 as cv
import numpy as np
import time

from Queue import Queue
from threading import Thread

import VARtracker


def main():
    print('VARframe(../video_carlos/, 1000, [[140, 170]], [[300, 500]])')
    VARmethod('../video_carlos/', 1000, [[140, 170]], [[300, 500]])
    print('VARframe(../video_tennis/, 1000, [[405, 160]], [[450, 275]])')
    VARmethod('../video_tennis/', 1000, [[405, 160]], [[450, 275]])
    print('VARframe(../video_tennis/, 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])')
    VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])
    print('VARframe(../video_tennis/, 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])')
    VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])


def addToQueue(queue, CMT, image):
    res = VARtracker.process_frame(CMT, image)
    queue.put(res)


def VARmethod(folder_path, final_frame, top_left, bottom_right):
    tic = time.time()

    if len(top_left) == len(bottom_right):
        list_cmt = [VARtracker.CMT() for index in range(len(top_left))]
        list_frame = [index for index in range(1, final_frame + 1)]
        list_name = [str(index) + '.jpg' for index in list_frame]

        # print ('using', top_left, bottom_right, 'as init bb')

        frame_path = folder_path + '/' + list_name[0]
        image0 = cv.imread(frame_path)
        gray0 = cv.cvtColor(image0, cv.COLOR_BGR2GRAY)

        for index in range(len(list_cmt)):
            VARtracker.initialise(list_cmt[index], gray0, top_left[index], bottom_right[index])

        frame_id = 1
        while frame_id < len(list_frame):
            frame_path = folder_path + '/' + list_name[frame_id]
            image = cv.imread(frame_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # copy = np.copy(image)

            queue = Queue()
            thread_list = []
            for cmt in list_cmt:
                line = Thread(target=addToQueue, args=(queue, cmt, gray))
                line.setDaemon(True)
                thread_list.append(line)

            [line.start() for line in thread_list]
            [line.join() for line in thread_list]
            # queue.join()

            while queue.qsize() > 0:
                # print('queue: ', queue.qsize())
                res = queue.get()
                if res.has_result:
                    cv.line(image, res.tl, res.tr, (255, 0, 0), 4)
                    cv.line(image, res.tr, res.br, (255, 0, 0), 4)
                    cv.line(image, res.br, res.bl, (255, 0, 0), 4)
                    cv.line(image, res.bl, res.tl, (255, 0, 0), 4)

            cv.imshow('main', image)
            cv.waitKey(1)
            # print(frame_id)
            frame_id += 1

    toc = time.time()
    print (toc - tic)


if __name__ == "__main__":
    main()
