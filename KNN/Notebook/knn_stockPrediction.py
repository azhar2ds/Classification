import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf

df=pdr.get_data_yahoo()
df.head()