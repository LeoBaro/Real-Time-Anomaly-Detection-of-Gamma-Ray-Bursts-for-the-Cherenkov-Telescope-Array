import os
import glob
import argparse
import os.path, time
from pathlib import Path

# ...

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str)
    args = parser.parse_args()

    handle = open("to_delete.txt", "w")

    counter = 0
    delcounter = 0
    for filename in glob.iglob( os.path.join(args.dir, '*.fits') ):
        
        counter += 1

        creationDate = time.ctime(os.path.getctime(Path(args.dir).joinpath(filename)))

        day, month, dayn, hour, year = creationDate.split(" ")

        if int(dayn) == 21:

            h,m,s = hour.split(":")

            if int(h) > 10 and int(m) > 50:
                delcounter += 1
                handle.write(str(Path(args.dir).joinpath(filename))+" "+creationDate+"\n")
        
        if counter % 1000 == 0:
            print("Progress...",counter)
    print(f"Delete {delcounter}/{counter}")