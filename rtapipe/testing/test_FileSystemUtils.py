import pytest
from pathlib import Path
from rtapipe.lib.rtapipeutils.FileSystemUtils import FileSystemUtils

class TestFileSystemUtils:

    def test_getAllFiles(self):

        testPath = Path(__file__).parent.joinpath("data", "tmp")

        testPath.mkdir(parents=True, exist_ok=True)

        fileNames = [f"tmp_{i}.fits" for i in range(3)] + \
                    [f"tmp_{i}.yaml" for i in range(3)] + \
                    [f"tmp_{i}.log" for i in range(3)]

        for f in fileNames:
            filePath = testPath.joinpath(f)
            with open(str(filePath), "w") as ff:
                ff.write(" ")

        files = FileSystemUtils.getAllFiles(testPath)

        assert len(files) == 3

        for f in files:
            assert ".fits" in f.name
            assert ".log" not in f.name
            assert ".yaml" not in f.name
