import cv2 as cv
import multiprocessing as mp
import numpy as np
import time

from functools import partial
from threading import Thread

import VARtracker


queue = mp.Queue()


def main():
    print('VARpool(../video_carlos/, 1000, [[140, 170]], [[300, 500]])')
    VARmethod('../video_carlos/', 1000, [[140, 170]], [[300, 500]])
    # print('VARpool(../video_tennis/, 1000, [[405, 160]], [[450, 275]])')
    # VARmethod('../video_tennis/', 1000, [[405, 160]], [[450, 275]])
    # print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])')
    # VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])
    # print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])')
    # VARmethod('../video_tennis/', 100, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])


def my_function(cmts, image):
    # print '{} running func with arg {}'.format(mp.current_process().name)
    # result = VARtracker.process_frame(cmts, image)
    print 'Entrou'
    return 1


def func(x,y,c):
    print '{} running func with arg {} and {}'.format(mp.current_process().name, x, y)
    return x


# def func(x, cmts, image):
#     print '{} running func with arg {}'.format(mp.current_process().name, x)
#     # VARtracker.process_frame(cmts, image)
#     return x


def VARmethod(folder_path, final_frame, top_left, bottom_right):
    tic = time.time()

    queue = mp.Queue()
    def callback(x):
        # print '{} running callback with arg'.format(mp.current_process().name,)
        queue.put(x)

    if len(top_left) == len(bottom_right):
        list_cmt = [VARtracker.CMT() for _ in range(len(top_left))]
        list_frame = [index for index in range(1, final_frame + 1)]
        list_name = [str(index) + '.jpg' for index in list_frame]

        frame_path = folder_path + '/' + list_name[0]
        image0 = cv.imread(frame_path)
        gray0 = cv.cvtColor(image0, cv.COLOR_BGR2GRAY)

        for index in range(len(list_cmt)):
            VARtracker.initialise(list_cmt[index], gray0, top_left[index], bottom_right[index])

        # pool = mp.Pool()

        frame_id = 1
        while frame_id < len(list_frame):
            frame_path = folder_path + '/' + list_name[frame_id]
            image = cv.imread(frame_path)
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            counter = 1
            for cmt in list_cmt:
                # pool.apply_async(func, args=(frame_id,counter,cmt), callback=callback)
                # pool.apply_async(my_function, args=(cmt, gray), callback=callback)
                counter += 1
                pool = mp.Process


            frame_id += 1

        pool.close()
        pool.join()

        print queue.qsize()

        print 'Finished with the script'

    toc = time.time()
    print (toc - tic)


if __name__ == "__main__":
    main()
