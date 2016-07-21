import copy_reg
import cv2 as cv
import logging
import multiprocessing as mp
import numpy as np
import Queue
import time
import types

import VARtracker


# def _reduce_method(m):
#     if m.im_self is None:
#         return getattr, (m.im_class, m.im_func.func_name)
#     else:
#         return getattr, (m.im_self, m.im_func.func_name)
# copy_reg.pickle(types.MethodType, _reduce_method)


def main():
    # print('VARpool(../video_carlos/, 1000, [[140, 170]], [[300, 500]])')
    # VARmethod('../video_carlos/', 1000, [[140, 170]], [[300, 500]])
    # print('VARpool(../video_tennis/, 1000, [[405, 160]], [[450, 275]])')
    # VARmethod('../video_tennis/', 1000, [[405, 160]], [[450, 275]])
    # print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])')
    # VARmethod('../video_tennis/', 1000, [[405, 160],[255, 100]], [[450, 275],[275, 155]])
    print('VARpool(../video_tennis/, 1000, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])')
    VARmethod('../video_tennis/', 200, [[405, 160],[255, 100],[340,80]], [[450, 275],[275, 155],[355,115]])


def worker(folder_path, list_name, top_left, bot_right):
    frame_path = folder_path + '/' + list_name[0]
    image_0 = cv.imread(frame_path)
    gray_0 = cv.cvtColor(image_0, cv.COLOR_BGR2GRAY)

    cmt = VARtracker.CMT()
    cmt.initialise(gray_0, top_left, bot_right)

    for name in list_name:
        frame_path = folder_path + '/' + name
        image_now = cv.imread(frame_path)
        gray_now = cv.cvtColor(image_now, cv.COLOR_BGR2GRAY)

        cmt.process_frame(gray_now)
        print 'Iterando'
    print 'Saiu'


queue = mp.Queue()
def on_return(result):
    queue.put(result)
    print 'Saiu'


def VARmethod(folder_path, final_frame, top_left, bot_right):
    tic = time.time()

    if len(top_left) == len(bot_right):
        list_frame = [index for index in range(1, final_frame + 1)]
        list_name = [str(index) + '.jpg' for index in list_frame]

        pool = mp.Pool(3)
        for item in zip(top_left, bot_right):
            pool.apply_async(worker, args=(folder_path, list_name, item[0], item[1]))
        pool.close()
        pool.join()


        print 'Finished with the script'

    toc = time.time()
    print (toc - tic)


if __name__ == "__main__":
    # mp.log_to_stderr(logging.DEBUG)
    main()
