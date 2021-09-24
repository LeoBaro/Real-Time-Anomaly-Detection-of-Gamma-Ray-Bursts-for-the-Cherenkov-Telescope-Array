import numpy as np
from numpy.core.shape_base import stack

class WindowsExtractor:

    def __init__(self):
        pass
    
    @staticmethod
    def test_extract_sub_windows(array, start, stop, sub_window_size, stride_size):
        """This method will extract subwindows from the axis=0 of a numpy array. 
        Check test_WindowsExtractor.py for examples.

        :param array: the input array, it can have different kind of shapes: [1,2,3], [[1],[2],[3]], [[1,1],[2,2],[3,3]]
        :type array: numpy.ndarray, required
        :param start: Start index 
        :type start: int, required
        :param stop: Stop index. A subwindow will never include the stop index.
        :type stop: int, required
        :param sub_window_size: the number of elements of the subwindows
        :type sub_window_size: int, required
        :param stride_size: each subwindows will be distant from the previous one by 'stride_size' 
        :type stride_size: int, required

        :return: Object of :class:`numpy.ndarray`
        :rtype: numpy.ndarray
        """
        assert start >= 0 and start < stop
        assert stop <= array.shape[0]
        assert sub_window_size > 0 and sub_window_size <= stop-start
        assert stride_size > 0

        #print("First subwindow indexes:",start + np.expand_dims(np.arange(sub_window_size), 0))# [[1 2 3]]
        #print("Slider:",np.expand_dims(np.arange(stop - start - sub_window_size + 1, step=stride_size), 0).T) #[[0], [1], .. , [10]]

        sub_windows = (
            start + 
            np.expand_dims(np.arange(sub_window_size), 0) +
            np.expand_dims(np.arange(stop - start - sub_window_size + 1, step=stride_size), 0).T
        )
        # print("sub_windows:",sub_windows)
        return array[sub_windows]    

  
    @staticmethod
    def test_extract_sub_windows_pivot(array, sub_window_size, stride_size, pivot_idx, delta=(-1,-1)):
        """This method will extract subwindows from the axis=0 of a numpy array. Some subwindows will be extracted
        before the pivot index, the other extracted subwindows can contain the pivot index. 
        Check test_WindowsExtractor.py for examples.

        :param array: the input array, it can have different kind of shapes: [1,2,3], [[1],[2],[3]], [[1,1],[2,2],[3,3]]
        :type array: numpy.ndarray, required
        :param sub_window_size: the number of elements of the subwindows
        :type sub_window_size: int, required
        :param stride_size: each subwindows will be distant from the previous one by 'stride_size' 
        :type stride_size: int, required
        :param pivot_idx: The index of the pivot element.
        :type pivot_idx: int, required
        :param delta: The number of subwindows to extract before and after the pivot (total=2*delta)
        :type delta: tuple(int,int), required

        :return: Object of :class:`numpy.ndarray`
        :rtype: numpy.ndarray
        """
        #num_subwindows_before_pivot = pivot_idx - sub_window_size # 0 1 2 3 4 5   pivot=4 sws=2 => num=2
        #num_subwindows_after_pivot = stop - pivot_idx + 1

        windows_before_pivot = WindowsExtractor.test_extract_sub_windows(array, 0, pivot_idx, sub_window_size, stride_size)
        windows_after_pivot = WindowsExtractor.test_extract_sub_windows(array, pivot_idx-sub_window_size+1, array.shape[0], sub_window_size, stride_size)        

        return (windows_before_pivot,windows_after_pivot)