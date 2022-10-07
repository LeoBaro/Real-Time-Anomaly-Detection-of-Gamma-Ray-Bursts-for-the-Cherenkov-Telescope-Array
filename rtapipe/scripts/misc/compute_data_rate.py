import argparse
import subprocess
import collections
from time import sleep, time
from datetime import datetime
from statistics import mean, stdev

def count_file():
    n_str = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.readlines()[0].decode('utf-8').strip()
    return int(n_str), time()

def now():
    return datetime.now().strftime('%d/%m/%Y_%H:%M:%S')

if __name__=='__main__':
    # count the number of files in a large directory

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', type=str, required=False, default="/scratch/baroncelli/DATA/obs/backgrounds_prod5b_10mln/run0406_ID000126")
    parser.add_argument('-s', '--sleep-sec', type=int, required=False, default=60)
    args = parser.parse_args()
    cmd = f"find {args.directory} -maxdepth 1 -name '*.fits' -type f | wc -l"
    sleep_t = args.sleep_sec

    n_prev = 0
    rates = collections.deque(maxlen=10)
    while(True):
        c_p_1, start_t = count_file()
        sleep(sleep_t)
        c_p_2, stop_t = count_file()
        produced = c_p_2 - c_p_1
        # compute rate
        rate = produced / (stop_t - start_t)
        rates.append(rate)
        if len(rates) > 1:
            print(f"[{now()}] n = {c_p_2} ({mean(rates):.2f}+-{stdev(rates):.2f} files/s)", end="\r")

