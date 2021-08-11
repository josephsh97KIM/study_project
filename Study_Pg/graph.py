import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pickle

with open('test.pickle', 'rb') as f:
    drowsy = pickle.load(f)
    yawns = pickle.load(f)

today = datetime.date.today()
today_day = str(today.day)
today_month = str(today.month)
today_date = today_month + '/' + today_day

data=[[today_date,yawns,drowsy]]

df=pd.DataFrame(data,columns=["Date","yawns","drowsy"])
df.plot(x="Date", y=["yawns","drowsy"], kind="bar",figsize=(4,6),title ="Detection",rot = 0)

plt.show()