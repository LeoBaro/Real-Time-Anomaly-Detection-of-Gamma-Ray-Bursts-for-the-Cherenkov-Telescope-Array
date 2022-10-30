from pathlib import Path

def parse_params(filename):
    try:
        params = {
            "runid":   filename.split("runid_")[1].split("_trial_")[0],
            "trial":   filename.split("trial_")[1].split("_simtype_")[0],
            "simtype": filename.split("simtype_")[1].split("_onset_")[0],
            "onset":   int(filename.split("onset_")[1].split("_delay_")[0]),
            "delay":   float(filename.split("delay_")[1].split("_offset_")[0]),
            "offset":  float(filename.split("offset_")[1].split("_itype_")[0]),
            "itype":   filename.split("itype_")[1].split("_itime_")[0],
            "itime":   int(filename.split("itime_")[1].split("_normalized_")[0]),
            "normalized": filename.split("normalized_")[1].split(".csv")[0]
        }
    except Exception as e:
        print("Fiilename that causes the error: ", filename)
        raise e
    return params
    
class FileSystemUtils:

    @staticmethod
    def getAllFiles(directory):
        dirpath = Path(directory)
        assert(dirpath.is_dir())
        fileList = []
        for x in dirpath.iterdir():
            if x.is_file() and ".yaml" not in x.name and ".log" not in x.name and "IRF" not in x.name:
                fileList.append(x)
        return fileList

    @staticmethod
    def iterDir(directory):
        dirpath = Path(directory)
        assert(dirpath.is_dir())
        for x in dirpath.iterdir():
            if x.is_file() and ".yaml" not in x.name and ".log" not in x.name and "IRF" not in x.name and ".yml" not in x.name:
                yield x

    @staticmethod
    def iterDirBatch(directory, batchSize):
        it = FileSystemUtils.iterDir(directory)
        if not it:
            raise StopIteration()
        while True:
            try:
                batch = []
                for x in range(batchSize):
                    batch.append(next(it))
                yield batch
            except:
                it = None
                break
        yield batch
