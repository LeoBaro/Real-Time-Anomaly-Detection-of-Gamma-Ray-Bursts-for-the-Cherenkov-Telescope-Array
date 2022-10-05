import argparse
import subprocess
from time import sleep, time

def count_file():
    n_str = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.readlines()[0].decode('utf-8').strip()
    return int(n_str), time()


if __name__=='__main__':
    # count the number of files in a large directory

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str, required=False, default="/scratch/baroncelli/DATA/obs/backgrounds_prod5b_10mln/run0406_ID000126")
    args = parser.parse_args()
    cmd = f"find {args.folder} -maxdepth 1 -name '*.fits' -type f | wc -l"

    sleep_t = 60
    n_prev = 0
    while(True):
        c_p_1, start_t = count_file()
        sleep(sleep_t)
        c_p_2, stop_t = count_file()
        produced = c_p_2 - c_p_1
        # compute rate
        rate = produced / (stop_t - start_t)
        print(f"n = {c_p_2} ({rate:.2f} files/s)")

