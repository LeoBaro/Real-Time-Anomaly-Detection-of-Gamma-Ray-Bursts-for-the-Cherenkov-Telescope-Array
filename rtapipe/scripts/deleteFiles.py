import os
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    args = parser.parse_args()

    with open(args.file, "r") as handle:

        files = handle.read().strip().split("\n")

        files = [file.split(" ")[0] for file in files]

        for f in files:
            os.remove(f)
            print(f, "removed..")
