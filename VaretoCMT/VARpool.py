import cv2 as cv
import itertools as it
import numpy as np
import time

from multiprocessing.dummy import Pool as ThreadPool
from Queue import Queue
from threading import Thread

import VARtracker


def main():
    # VARmethod('../video_carlos/', 200, [[140, 170]], [[300, 500]])
    # VARmethod('../video_tennis/', 200, [[405, 160]], [[450, 275]])
    VARmethod('../video_tennis/', 200, [[405, 160],[255, 100]], [[450, 275],[275, 155]])


response_list = []
def addToQueue(result):
    response_list.append(result)


def calculateParallel(cmts, image):
    pool = ThreadPool(6)
    queue = Queue()
    temp = zip(cmts, image)
    # results = pool.map(VARtracker.process_frame, zip(cmts, image))
    for cmt in cmts:
        result = pool.apply_async(VARtracker.process_frame, args=(cmt, image), callback=addToQueue)
    pool.close()
    pool.join()
    return queue


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


            # thread_list = []
            # for cmt in list_cmt:
            #     line = Thread(target=addToQueue, args=(queue, cmt, gray))
            #     line.setDaemon(True)
            #     thread_list.append(line)
            #
            # [line.start() for line in thread_list]
            # [line.join() for line in thread_list]

            queue = calculateParallel(list_cmt, gray)

            for response in response_list:
                if response.has_result:
                    cv.line(image, response.tl, response.tr, (255, 0, 0), 4)
                    cv.line(image, response.tr, response.br, (255, 0, 0), 4)
                    cv.line(image, response.br, response.bl, (255, 0, 0), 4)
                    cv.line(image, response.bl, response.tl, (255, 0, 0), 4)

            cv.imshow('main', image)
            cv.waitKey(1)
            print(frame_id)
            frame_id += 1

    toc = time.time()
    print (toc - tic)


if __name__ == "__main__":
    main()
