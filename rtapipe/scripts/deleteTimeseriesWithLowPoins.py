import os
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from rtapipe.lib.rtapipeutils.FileSystemUtils import FileSystemUtils


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--data-dir", type=str, required=True, help="")
    parser.add_argument("-es", "--expeceted-size", type=int, required=True, help="")
    args = parser.parse_args()

    command = f"ls -afq {args.data_dir} | wc -l"
    counts = int(os.popen(command).read())

    dirIterator = FileSystemUtils.iterDir(Path(args.data_dir))

    with open("./to_delete_files.txt", "w") as handle:

        for i in tqdm(range(counts)):

            try:
                file = next(dirIterator)
                try:
                    bf = pd.read_csv(file, sep=",")
                except Exception as genericEx:
                    print(f"Problem in {file} with exception: {genericEx}")
                    handle.write(f"\n{file}")
                except pandas.errors.EmptyDataError:
                    print(f"Problem in {file} with exception: pandas.errors.EmptyDataError")
                    handle.write(f"\n{file}")
                finally:
                    if(bf.shape[0] != args.expeceted_size):
                        print(f"Problem in {file} with shape {bf.shape}")
                        handle.write(f"\n{file}")
            except StopIteration:
                print("StopIteration exception!")
                break
