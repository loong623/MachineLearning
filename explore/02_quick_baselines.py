import sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
df = pd.read_csv(sys.argv[1], parse_dates=["Date"]).sort_values(["Location","Date"])
zone = {}
for z, locs in {"Tropical":"Darwin Katherine Cairns Townsville",
 "Arid":"AliceSprings Uluru Woomera Cobar Mildura SalmonGums Moree",
 "Subtropical":"Brisbane GoldCoast CoffsHarbour Sydney SydneyAirport Newcastle Williamtown NorahHead Wollongong NorfolkIsland Richmond Penrith BadgerysCreek",
 "Mediterranean":"Perth PerthAirport PearceRAAF Adelaide Nuriootpa Albany Walpole Witchcliffe",
 "Temperate":"Melbourne MelbourneAirport Watsonia Ballarat Bendigo Sale Portland Dartmoor MountGambier Nhil Hobart Launceston Canberra Tuggeranong WaggaWagga Albury MountGinini"}.items():
    for l in locs.split(): zone[l]=z
df["Zone"]=df.Location.map(zone); print("unmapped", df.Zone.isna().sum()); print((df.Zone.value_counts(normalize=True)*100).round(1).to_dict())
num = ["MinTemp","MaxTemp","Rainfall","WindGustSpeed","WindSpeed9am","WindSpeed3pm","Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm"]
g = df.groupby("Location")
df["y_frost"]=(g.MinTemp.shift(-1)<=2); df["y_hot"]=(g.MaxTemp.shift(-1)>=35); df["y_gale"]=(g.WindGustSpeed.shift(-1)>=63)
df["y_cool"]=((g.MaxTemp.shift(-1)-df.MaxTemp)<=-8); df["y_rain"]=df.RainTomorrow.eq("Yes")
d = df.dropna(subset=num).sample(40000, random_state=0)
X = d[num].values; X=(X-X.mean(0))/X.std(0)
def run(name,y,multi=False):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=0,stratify=y)
    for m in [LogisticRegression(max_iter=2000), DecisionTreeClassifier(max_depth=8,random_state=0)]:
        m.fit(Xtr,ytr); p=m.predict(Xte)
        if multi: s=f"acc={accuracy_score(yte,p):.3f} macroF1={f1_score(yte,p,average='macro'):.3f}"
        else: s=f"AUC={roc_auc_score(yte,m.predict_proba(Xte)[:,1]):.3f} F1={f1_score(yte,p):.3f}"
        print(f"{name:10s} {type(m).__name__[:8]} {s}")
for t in ["y_rain","y_frost","y_hot","y_gale","y_cool"]: run(t, d[t].values.astype(int))
run("zone", d.Zone.values, True)
season = d.Date.dt.month.map(lambda m:{12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}[m]).values
run("season", season, True)
