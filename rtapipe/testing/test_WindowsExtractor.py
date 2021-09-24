import pytest
import numpy as np
from rtapipe.lib.rtapipeutils.WindowsExtractor import WindowsExtractor

class TestWindowsExtractor:

    def assert_array_equal(arr1, arr2):
        assert arr1.shape == arr2.shape
        arr1 = arr1.flatten()
        arr2 = arr2.flatten()
        for idx in range(len(arr1)):
            assert arr1[idx] == arr2[idx]
        

    @pytest.mark.parametrize("test_input,expected_windows", 
    [
        ({"arr":np.arange(10,20),"start":0,"stop":10,"ws":3,"stride":1}, np.array([[10,11,12],[11,12,13],[12,13,14],[13,14,15],[14,15,16],[15,16,17],[16,17,18],[17,18,19]])),
        ({"arr":np.arange(10,20),"start":3,"stop":10,"ws":3,"stride":1}, np.array([[13,14,15],[14,15,16],[15,16,17],[16,17,18],[17,18,19]])),
        ({"arr":np.arange(10,20),"start":3,"stop":7,"ws":3,"stride":1}, np.array([[13,14,15],[14,15,16]])),
        ({"arr":np.arange(10,20),"start":2,"stop":8,"ws":3,"stride":2}, np.array([[12,13,14],[14,15,16]])),
        ({"arr":np.arange(10,20).reshape(10,1),"start":0,"stop":10,"ws":3,"stride":1}, np.expand_dims(np.array([[10,11,12],[11,12,13],[12,13,14],[13,14,15],[14,15,16],[15,16,17],[16,17,18],[17,18,19]]), axis=2)),
        ({"arr":np.arange(10,20).reshape(10,1),"start":3,"stop":10,"ws":3,"stride":1}, np.expand_dims(np.array([[13,14,15],[14,15,16],[15,16,17],[16,17,18],[17,18,19]]), axis=2)),
        ({"arr":np.arange(10,20).reshape(10,1),"start":3,"stop":7,"ws":3,"stride":1}, np.expand_dims(np.array([[13,14,15],[14,15,16]]), axis=2)),
        ({"arr":np.arange(10,20).reshape(10,1),"start":2,"stop":8,"ws":3,"stride":2}, np.expand_dims(np.array([[12,13,14],[14,15,16]]), axis=2)),
        ({"arr":np.repeat(np.arange(10,20), 2).reshape(10,2),"start":0,"stop":10,"ws":3,"stride":1}, np.array([  [[10,10],[11,11],[12,12]], [[11,11],[12,12],[13,13]], [[12,12],[13,13],[14,14]], [[13,13],[14,14],[15,15]], [[14,14],[15,15],[16,16]],  [[15,15],[16,16],[17,17]], [[16,16],[17,17],[18,18]], [[17,17],[18,18],[19,19]]   ])),
        ({"arr":np.repeat(np.arange(10,20), 2).reshape(10,2),"start":3,"stop":10,"ws":3,"stride":1}, np.array([  [[13,13],[14,14],[15,15]], [[14,14],[15,15],[16,16]],  [[15,15],[16,16],[17,17]], [[16,16],[17,17],[18,18]], [[17,17],[18,18],[19,19]]   ])),
        ({"arr":np.repeat(np.arange(10,20), 2).reshape(10,2),"start":3,"stop":7,"ws":3,"stride":1}, np.array([  [[13,13],[14,14],[15,15]], [[14,14],[15,15],[16,16]]   ])),
        ({"arr":np.repeat(np.arange(10,20), 2).reshape(10,2),"start":2,"stop":8,"ws":3,"stride":2}, np.array([  [[12,12],[13,13],[14,14]], [[14,14],[15,15],[16,16]]   ])),
    ])

    def test_extract_sub_windows(self,test_input,expected_windows):
        windows = WindowsExtractor.test_extract_sub_windows(test_input["arr"], test_input["start"], test_input["stop"], test_input["ws"], test_input["stride"])
        TestWindowsExtractor.assert_array_equal(windows, expected_windows)

    # 10,11,12,13,14,15,16,17,18,19
    #      piv=2

    @pytest.mark.parametrize("test_input,expected_windows", 
    [
        ({"arr":np.arange(10,20),"sub_window_size":3,"stride":1,"pivot_idx":5}, (np.array([[10,11,12], [11,12,13], [12,13,14]]), np.array([[13,14,15],[14,15,16],[15,16,17],[16,17,18],[17,18,19]]))),
        ({"arr":np.arange(10,20),"sub_window_size":3,"stride":2,"pivot_idx":5}, (np.array([[10,11,12], [12,13,14]]), np.array([[13,14,15],[15,16,17],[17,18,19]]))),
        ({"arr":np.arange(10,20),"sub_window_size":2,"stride":1,"pivot_idx":2}, (np.array([[10,11]]), np.array([[11,12],[12,13],[13,14],[14,15],[15,16],[16,17],[17,18],[18,19]])))
    ])

    def test_extract_sub_windows_pivot(self,test_input,expected_windows):
        windows = WindowsExtractor.test_extract_sub_windows_pivot(test_input["arr"],test_input["sub_window_size"], test_input["stride"], test_input["pivot_idx"])
        TestWindowsExtractor.assert_array_equal(windows[0], expected_windows[0])
        TestWindowsExtractor.assert_array_equal(windows[1], expected_windows[1])

    




    