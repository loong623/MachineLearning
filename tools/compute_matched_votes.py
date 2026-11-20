import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
z=np.load('results/predictions.npz', allow_pickle=False)
station=z['station']; date=pd.to_datetime(z['date'])
sy=z['seasonal_station']; years=z['seasonal_year']; y=z['seasonal_y']
labels=['Tropical','Arid','Subtropical','Mediterranean','Oceanic']
specs=[(3,20),(6,21),(9,22),(12,21)]
rows=[]
for model, key, display in [('LR','daily_LR_s42','daily LR majority over four nearest seasonal snapshots'),('HGB','daily_HGB_s42','daily HGB majority over four nearest seasonal snapshots')]:
    pred=[]
    for s,year in zip(sy,years):
        inds=np.flatnonzero((station==s)&(date.year==year)); picks=[]
        for month,day in specs:
            centre=pd.Timestamp(year=int(year),month=month,day=day)
            delta=np.abs((date[inds]-centre).days)
            valid=delta<=10
            if valid.any(): picks.append(inds[np.argmin(np.where(valid,delta,9999))])
        votes=pd.Series(z[key][picks]).value_counts()
        pred.append(sorted(votes.index,key=lambda q:(-votes[q],labels.index(q)))[0])
    rows.append({'representation':display,'rows':len(y),'macro_F1':f1_score(y,pred,labels=labels,average='macro'),'accuracy':accuracy_score(y,pred)})
rows += [
 {'representation':'seasonal HGB-tuned signature','rows':len(y),'macro_F1':f1_score(y,z['seasonal_HGB-tuned_s42'],labels=labels,average='macro'),'accuracy':accuracy_score(y,z['seasonal_HGB-tuned_s42'])},
 {'representation':'seasonal LR signature','rows':len(y),'macro_F1':f1_score(y,z['seasonal_LR_s42'],labels=labels,average='macro'),'accuracy':accuracy_score(y,z['seasonal_LR_s42'])},
]
pd.DataFrame(rows).to_csv('results/seasonal_matched_baseline.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
