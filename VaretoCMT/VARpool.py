import cv2 as cv

import numpy as np
import time

from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from Queue import Queue
from threading import Thread

import VARtracker

def main():
    print('VARpool(../video_carlos/, 1000, [[140, 170]], [[300, 500]])')
    VARmethod('../video_carlos/', 1000, [[140, 170]], [[300, 500]])
    print('VARpool(../video_tennis/, 1000, [[405, 160]], [[450, 275]])')
    VARmethod('../video_tennis/', 1000, [[405, 160]], [[450, 275]])
    print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])')
    VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])
    print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])')
    VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])


def calculate_parallel(image, cmts):
    results = VARtracker.process_frame(cmts, image)
    return results


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

            pool = ThreadPool(3)
            method = partial(calculate_parallel, gray)
            response_list = pool.map(method, list_cmt)
            # pool.map_async(method, list_cmt)
            pool.close()
            pool.join()

            for response in response_list:
                if response.has_result:
                    cv.line(image, response.tl, response.tr, (255, 0, 0), 4)
                    cv.line(image, response.tr, response.br, (255, 0, 0), 4)
                    cv.line(image, response.br, response.bl, (255, 0, 0), 4)
                    cv.line(image, response.bl, response.tl, (255, 0, 0), 4)

            cv.imshow('main', image)
            cv.waitKey(1)
            # print(frame_id)
            frame_id += 1

    toc = time.time()
    print (toc - tic)


if __name__ == "__main__":
    main()
