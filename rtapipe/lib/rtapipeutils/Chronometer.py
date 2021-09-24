from time import time
import numpy as np

class Chronometer:

    def __init__(self):
        self.start_time = None
        self.elapsed = []
        self.total_time = 0

    def start(self):
        self.start_time = time()

    def stop(self):
        now = time()
        elapsed = now-self.start_time
        self.elapsed.append(elapsed)
        self.total_time += elapsed

    def get_statistics(self):
        return np.array(self.elapsed).mean(),np.array(self.elapsed).std()

    def get_total_elapsed_time(self):
        return self.total_time

    def reset(self):
        self.start_time = None
        self.elapsed = []    

"""
if __name__=='__main__':
    from time import sleep

    c = Chronometer()

    c.start()
    sleep(3)
    c.stop()

    c.start()
    sleep(2.5)
    c.stop()

    c.start()
    sleep(3.5)
    c.stop()

    mean,dev = c.get_statistics()

    print(c.elapsed)
    print(mean,dev)
"""