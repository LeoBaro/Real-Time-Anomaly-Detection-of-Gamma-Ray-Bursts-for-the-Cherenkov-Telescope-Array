import numpy as np

def extract_windows(array, start, max_time, sub_window_size):
    examples = []
    stop = max_time-sub_window_size-start+1
    for i in range(stop):
        example = array[start+i:start+sub_window_size+i]
        examples.append(np.expand_dims(example, 0))
    
    print(examples)
    print(len(examples))
    return np.vstack(examples)
    



if __name__ == "__main__":

    TMAX = 10
    WS=5
    arr = np.array([i for i in range(TMAX)])
    arr = np.random.randint(0, 5, size=(10,2))
    
    print(arr)
    print(extract_windows(arr, 0, TMAX, WS))
