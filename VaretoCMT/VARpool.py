import cv2 as cv
import logging
import multiprocessing as mp
import numpy as np
import Queue
import time

from functools import partial
from threading import Thread

import VARtracker


# queue = mp.Queue()


def main():
    # print('VARpool(../video_carlos/, 1000, [[140, 170]], [[300, 500]])')
    # VARmethod('../video_carlos/', 1000, [[140, 170]], [[300, 500]])
    print('VARpool(../video_tennis/, 1000, [[405, 160]], [[450, 275]])')
    VARmethod('../video_tennis/', 100, [[405, 160]], [[450, 275]])
    # print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])')
    # VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])
    # print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])')
    # VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])


def worker(image_0, image_now, top_left, bot_right):
    cmt = VARtracker.CMT()
    cmt.initialise(image_0, top_left, bot_right)
    result = cmt.process_frame(image_now)
    return result

queue = []
def on_return(result):
    # queue.put(result)
    queue.append(result)
    print 'Saiu'


def func(x,y):
    print '{} running func with arg {} and {}'.format(mp.current_process().name, x, y)
    return x


def VARmethod(folder_path, final_frame, top_left, bot_right):
    tic = time.time()

    if len(top_left) == len(bot_right):
        list_frame = [index for index in range(1, final_frame + 1)]
        list_name = [str(index) + '.jpg' for index in list_frame]

        frame_path = folder_path + '/' + list_name[0]
        image_0 = cv.imread(frame_path)
        gray_0 = cv.cvtColor(image_0, cv.COLOR_BGR2GRAY)

        pool = mp.Pool(1)

        frame_id = 1
        while frame_id < len(list_frame):

            frame_path = folder_path + '/' + list_name[frame_id]
            image_now = cv.imread(frame_path)
            gray_now = cv.cvtColor(image_now, cv.COLOR_BGR2GRAY)

            for index in range(len(top_left)):
                pool.apply_async(worker, args=(gray_0, gray_now, top_left[index], bot_right[index]), callback=on_return)

            print frame_id
            frame_id += 1

        pool.close()
        pool.join()

        print len(queue)

        print 'Finished with the script'

    toc = time.time()
    print (toc - tic)


if __name__ == "__main__":
    mp.log_to_stderr(logging.DEBUG)
    main()
