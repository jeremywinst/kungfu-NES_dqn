import os
import time

os.environ['CUDA_VISIBLE_DEVICES']='0'

def ConvertSectoDay(n):
    hour = n // 3600

    n %= 3600
    minutes = n // 60

    n %= 60
    seconds = n

    print(hour, "hours", minutes, "minutes", int(seconds), "seconds")

start_ALL = time.time()

os.system('python kungfu_dqn.py --name left15 --r_left 15 --episode 750 --lr 0.0001 --decay 200 --end 0.1')

elapsed_time_ALL = time.time() - start_ALL
print('\n\n\nTRAINING FINISH!')
print('TOTAL EXECUTION TIME: ', ConvertSectoDay(elapsed_time_ALL))