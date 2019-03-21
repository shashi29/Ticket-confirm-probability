
import pandas as pd
import numpy as np
import seaborn as sns

train = pd.read_csv("TRAIN.TXT",sep=' ')
train.reset_index(inplace=True)

train.rename(columns={'level_0': 'Date'}, inplace=True)
train['Date'] = pd.to_datetime(train['Date'])
train.sort_index(by='Date',ascending=False)

train['level_1'] = train.level_1.str.replace('00:00:00.000	', '')
train.rename(columns={'level_1': 'Booking_status'}, inplace=True)
train['Booking_status'].value_counts()
#TQWL 2398
#Wl 827

#drop 3rd column containing NAN only
train.drop(columns=['level_2'],axis=1,inplace=True)
train.fillna(0,inplace=True)

train.to_csv('temp.txt',index=None)


#Remove space with 
train['level_3'] = train.level_3.str.replace('nan',',')
train['level_3'] = train.level_3.str.replace(' ',',')

import re
original_string = open('TRAIN.TXT').read()
new_string = re.sub('[^a-zA-Z0-9\n\.\*\/\-]', ' ', original_string)
open('bar.txt', 'w').write(new_string)

with open('bar.txt', 'r') as f:
    lines = f.readlines()

# remove spaces
lines = [line.replace('  ', '') for line in lines]

# finally, write lines in the file
with open('file.txt', 'w') as f:
    f.writelines(lines)

#new = train["level_3"].str.split(",", n = 1, expand = True) 

#From this extarct train number information
#train['Train_no'] = new[1]
#train['Train_no'] = train.train.str.replace('', '')
#Update the value
#train['level_3'] = train.level_3.str.replace('TQ	*12864','')
#train['level_3'] = train.level_3.str.replace('12864','')

#train['level_34'] = train['level_3'].map(str) + train['level_4'] 

new_df = pd.read_csv("file.txt",sep=',')

data['ChartingStatus'] = data.ChartingStatus.str.replace('[^0-9]+','1')
