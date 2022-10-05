from pathlib import Path
from time import time, sleep
from multiprocessing import Pool

class APMockStreaming:

    def __init__(self, data_dir, data_rate) -> None:
        self.data_dir = data_dir
        self.data_rate = data_rate
        
    # iterator function that return a file from a large directory
    def get_file(self):
        print("APMockStreaming - get_file")
        start_t = time()
        for file in Path(self.data_dir).iterdir():
            if not file.is_file():
                continue
            while time() - start_t < self.data_rate:
                continue
            yield file
            start_t = time()

if __name__=='__main__':
    # Monitor a directory with 
    ap = APMockStreaming("/home/rtapipe/rtapipe/training/online/data", 2)
    for file in ap.get_file():
        print(file)
