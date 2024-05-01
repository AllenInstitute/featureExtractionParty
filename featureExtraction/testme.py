import pandas as pd
import numpy as np
import os
import tempfile
import fsspec


df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
print(df)


with fsspec.open('simplecache::gs://allen-minnie-phase3/PSS/test/test12345.pkl', 'wb',
                 gcs={'project': 'em-270621'}) as f:
    #f.write(b"some data")
    df.to_pickle(f)