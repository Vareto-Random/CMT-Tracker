from trackers import worker
import time

# jobC = worker.delay('../video_carlos/', 1, 1000, [140, 170], [300, 500], 1)

job1 = worker.delay('../video_tennis/', 1, 100, [405, 160], [450, 275], 1)
time.sleep(5)
job2 = worker.delay('../video_tennis/', 1, 100, [255, 100], [275, 155], 2)
time.sleep(5)
job3 = worker.delay('../video_tennis/', 1, 100, [340,  80], [355, 115], 3)
time.sleep(5)
job4 = worker.delay('../video_tennis/', 1, 100, [270,  80], [280, 103], 4)
time.sleep(5)
job5 = worker.delay('../video_tennis/', 1, 100, [245,  75], [253,  99], 5)

# HOW TO RUN
# Start celery: celery worker -A trackers &
# Python: from trackers import worker
# jobC = worker.delay('../video_carlos/', 1, 100, [140, 170], [300, 500], 1)
# Stop celery: ps auxww | grep 'celery worker' | awk '{print $2}'| xargs kill