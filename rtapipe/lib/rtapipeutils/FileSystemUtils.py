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