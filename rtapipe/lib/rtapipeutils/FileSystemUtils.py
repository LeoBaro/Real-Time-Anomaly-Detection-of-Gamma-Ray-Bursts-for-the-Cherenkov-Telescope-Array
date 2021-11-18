from pathlib import Path

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
