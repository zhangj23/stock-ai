To run, type `python model.py` to run the model.py file in the terminal. This file uses the `stock_classes.py` file where all the data transformation and model training

Test Results for LSTM without BiDirectional sequences and stock sentiment data with one year of data
Batch size 64
|seq_length| accuracy |
| -------- | ------- |
| 5 | 53.27% |
| 10 | 54% |
| 20 | 57.14% |

Test Results for LSTM without BiDirectional sequences and stock sentiment data with 6 months of data
Batch size 32
|seq_length| accuracy |
| -------- | ------- |
| 3 | 58.09% |
| 5 | 62.7% |
| 7 | 57.14% |

Test Results for LSTM without BiDirectional sequences and no stock sentiment data with 6 months of data
Batch size 32
|seq_length| accuracy | date |
| -------- | ------- | ------- |
| 3 | 68% | 01/21/2025 |
| 5 | 58.5% | 01/21/2025|
| 7 | 62.22% | 01/21/2025 |
| 3 | 68% | 01/24/2025 |
| 5 | 53.3% | 01/24/2025 |
| 7 | 48.88% | 01/24/2025 |
