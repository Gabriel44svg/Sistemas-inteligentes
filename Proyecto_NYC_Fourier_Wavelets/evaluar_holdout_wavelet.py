"""Una sola medicion holdout para el candidato elegido SOLO en validacion.

No correr para afinar pesos. No usa etiquetas holdout para preparar columnas.
"""
import __main__,json,gc
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from _mejora_geografia_v2 import Augment, LeakageFreeEncoder
from features_taxi import create_features
from wavelet_fft_taxi import deterministic_features,rmsle
__main__.Augment=Augment;__main__.LeakageFreeEncoder=LeakageFreeEncoder
R=Path(__file__).resolve().parent; O=R/'resultados_wavelet'
win=json.load(open(O/'ganador_final_validacion.json'))['mejor']
assert win['modelo']=='ensemble_previo_0.75_con_Fourier_demanda_target_OOF',win
df=pd.read_csv(R/'train.csv',parse_dates=['pickup_datetime']);_,tmp=train_test_split(np.arange(len(df)),test_size=.20,random_state=42);_,ih=train_test_split(tmp,test_size=.50,random_state=42)
ho=df.iloc[ih].copy().reset_index(drop=True)
print('Holdout intacto:',ho.shape,flush=True)
base_X=create_features(ho)
p=joblib.load(R/'mejora_taxis/v2/referencia_81_validacion.joblib')
x=base_X.to_numpy(dtype='float32');stats=p['imputer'].statistics_[:81].astype('float32');x=np.where(np.isfinite(x),x,stats).astype('float32')
zref=p['model'].predict(x);del p,x;gc.collect()
print('predict ref',flush=True)
p=joblib.load(R/'mejora_taxis/v2/rotacion_mas_oof_validacion.joblib')
fr=pd.concat([base_X.reset_index(drop=True),p['augment'].transform(ho).reset_index(drop=True)],axis=1)
e=p['encoder'].transform(fr);x=pd.concat([fr,e],axis=1)
cl=list(p['imputer'].feature_names_in_);xx=p['imputer'].transform(x[cl]);ix=[cl.index(col) for col in p['cols']]
zgeo=p['model'].predict(xx[:,ix]);del p,fr,e,x,xx;gc.collect()
print('predict geo oof',flush=True)
p=joblib.load(O/'wavelet_modelo_validacion.joblib')
x=pd.concat([base_X.reset_index(drop=True),deterministic_features(ho).reset_index(drop=True),p['encoder'].transform(ho).reset_index(drop=True)],axis=1)
assert list(x.columns)==p['columnas']
xx=p['imputer'].transform(x);zspec=p['modelo'].predict(xx[:,p['seleccion']]);del p,x,xx;gc.collect()
print('predict wavelet OOF',flush=True)
base=(zref+zgeo)/2
z=0.75*base+0.25*zspec
score=rmsle(ho.trip_duration,np.expm1(np.maximum(z,0)))
base_score=rmsle(ho.trip_duration,np.expm1(np.maximum(base,0)))
with open(O/'holdout_final.json','w') as f:
 json.dump({'ganador_validacion':win,'rmsle_holdout_ensamble_previo':base_score,
            'rmsle_holdout_wavelet':score,'holdout_n':len(ho),
            'regla':'NO optimizar pesos con holdout; test.csv nunca etiquetado',
            'mejora':base_score-score},f,indent=2)
print('FINAL HOLDOUT: PREV',base_score,'FOURIER + HAAR',score,'DELTA',base_score-score,flush=True)
