import pandas as pd
import numpy as np


def forgetting_table(acc_dataframe, n_tasks=5):
    return pd.DataFrame(    
        [forgetting_line(acc_dataframe, task_id=i).values[:,-1].tolist() for i in range(0, n_tasks)]
    )

def forgetting_line(acc_dataframe, task_id=4, n_tasks=5):
    if task_id == 0:
        forgettings = [np.nan] * n_tasks
    else:
        forgettings = [forgetting(task_id, p, acc_dataframe) for p in range(task_id)] + [np.nan]*(n_tasks-task_id)

    # Create dataframe to handle NaN
    return pd.DataFrame(forgettings)

def forgetting(q, p, df):
    D = {}
    for i in range(0, q+1):
        D[f"d{i}"] = df.diff(periods=-i)

    # Create datafrmae to handle NaN
    return pd.DataFrame(([D[f'd{k}'].iloc[q-k,p] for k in range(0, q+1)])).max()[0]