from celery import Celery

import cv2 as cv
import VARtracker

app = Celery('trackers', backend='amqp', broker='amqp://')
app.conf.CELERY_REDIRECT_STDOUTS=False

@app.task
def worker(folder_path, initial_frame, final_frame, top_left, bot_right, id):
    if len(top_left) == len(bot_right):
        list_frame = [index for index in range(initial_frame, final_frame + 1)]
        list_name = [str(index) + '.jpg' for index in list_frame]

        frame_path = folder_path + '/' + list_name[0]
        image_0 = cv.imread(frame_path)
        gray_0 = cv.cvtColor(image_0, cv.COLOR_BGR2GRAY)

        cmt = VARtracker.CMT()
        cmt.initialise(gray_0, top_left, bot_right)
        list_of_bbs = []

        for name in list_name:
            frame_path = folder_path + '/' + name
            image_now = cv.imread(frame_path)
            gray_now = cv.cvtColor(image_now, cv.COLOR_BGR2GRAY)

            cmt.process_frame(gray_now)
            if cmt.has_result:
                list_of_bbs.append((id, name, cmt.tl, cmt.br))
        print 'Processo {} terminou'.format(id)

        return list_of_bbs
    else:
        return None


# HOW TO RUN
# Start celery: celery worker -A trackers &
# Python: from trackers import worker
# jobC = worker.delay('../video_carlos/', 1, 100, [140, 170], [300, 500], 1)
# Stop celery: ps auxww | grep 'celery worker' | awk '{print $2}'| xargs kill
