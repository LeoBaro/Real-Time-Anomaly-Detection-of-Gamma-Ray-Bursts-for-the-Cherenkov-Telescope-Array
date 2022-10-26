import numpy as np
import tensorflow as tf
from pathlib import Path
from rtapipe.lib.evaluation.custom_mse import CustomMSE 

class TestCustomMSE:

    def test_mse(self):
                
        y_true = np.array([
            [ 
                [1, 2],
                [3, 4],
                [5, 6]
            ],
            [ 
                [10, 20],
                [30, 40],
                [50, 60]
            ]    
        ])

        y_pred = np.array([
            [ 
                [0, 0],
                [0, 0],
                [0, 0]
            ],
            [ 
                [0, 0],
                [0, 0],
                [0, 0]
            ]
            
        ])
        
        c_mse = CustomMSE(n_features=2, output_dir=Path(__file__).parent.joinpath("test_custom_mse_output"))

        loss = c_mse.call(tf.constant(y_true, dtype=tf.float32), tf.constant(y_pred, dtype=tf.float32))
        assert loss.numpy() == 707.0

        c_mse.write_reconstruction_errors()