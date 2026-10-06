from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
FEATURES=['ret_1d','ret_5d','ret_20d','ret_60d','vol_20d','vol_60d','price_sma10','price_sma20','price_sma50','price_sma100','rsi_14','range_pct','volume_z']

def walk_forward_ml(df, train_fraction=.7, threshold=.55, seed=42):
    x=df.dropna(subset=FEATURES+['target']).copy(); split=int(len(x)*train_fraction); train,test=x.iloc[:split],x.iloc[split:]
    model=RandomForestClassifier(n_estimators=400,max_depth=6,min_samples_leaf=10,class_weight='balanced_subsample',random_state=seed,n_jobs=-1)
    model.fit(train[FEATURES],train.target); pred=model.predict(test[FEATURES]); proba=model.predict_proba(test[FEATURES])[:,1]
    signal=pd.Series(0.,index=df.index); signal.loc[test.index]=(proba>=threshold).astype(float)
    cm=confusion_matrix(test.target,pred,labels=[0,1]); auc=roc_auc_score(test.target,proba) if test.target.nunique()==2 else np.nan
    stats={'Accuracy':accuracy_score(test.target,pred),'Precision':precision_score(test.target,pred,zero_division=0),'Recall':recall_score(test.target,pred,zero_division=0),'ROC AUC':auc,'Confusion':cm,'Feature Importance':pd.Series(model.feature_importances_,index=FEATURES).sort_values(ascending=False),'Train Rows':len(train),'Test Rows':len(test),'Predictions':test.assign(prediction=pred,prob_up=proba)}
    return signal,stats
