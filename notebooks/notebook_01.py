#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd

from energy_analysis import LineChart


# In[2]:


dir_path = Path("../data")
file_name = "energybalance_240320.csv"

file_path = dir_path / file_name


# In[3]:


df = pd.read_csv(file_path, sep=";", thousands=",")


# In[4]:


df.info()


# In[13]:


columns = {
    "time": 0,
    "direct consumption": 1,
    "battery discharging": 2,
    "grid import": 3,
    "total consumption": 4,
    "grid export": 5,
    "direct consumption (bis)": 6,
    "battery charging": 7,
    "PV-power": 8
}


# In[6]:


time_index = np.asarray([k for k in range(df.shape[0])])
time_index


# In[8]:


pv_power = df.iloc[:, columns["PV-power"]].to_numpy()
pv_power


# In[14]:


dir_consumption = df.iloc[:, columns["direct consumption"]].to_numpy()
dir_consumption


# In[16]:


surplus = pv_power - dir_consumption
surplus


# In[18]:


battery_charging = df.iloc[:, columns["battery charging"]].to_numpy()
battery_charging


# In[15]:


ch = LineChart()
ch.add_xy_data(
    label="PV-power",
    x1_values=time_index,
    y1_values=pv_power
)
ch.add_xy_data(
    label="direct consumption",
    x1_values=time_index,
    y1_values=dir_consumption
)
ch.x1.add_title("time index")
ch.y1.add_title("power, W")
ch.add_legend()
ch.show()


# In[19]:


ch = LineChart()
ch.add_xy_data(
    label="surplus",
    x1_values=time_index,
    y1_values=surplus
)
ch.add_xy_data(
    label="battery charging",
    x1_values=time_index,
    y1_values=battery_charging
)
ch.x1.add_title("time index")
ch.y1.add_title("power, W")
ch.add_legend()
ch.show()

