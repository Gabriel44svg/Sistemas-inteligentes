import __main__,joblib,numpy as np,pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from _mejora_geografia_v2 import Augment,LeakageFreeEncoder
from features_taxi import create_features
from wavelet_fft_taxi import rmsle
__main__.Augment=Augment; __main__.LeakageFreeEncoder=LeakageFreeEncoder
R=Path(__file__).resolve().parent; O=R/'resultados_wavelet'
d=pd.read_csv(R/'train.csv',parse_dates=['pickup_datetime']);_, temp=train_test_split(np.arange(len(d)),test_size=.2,random_state=42);vi,_=train_test_split(temp,test_size=.5,random_state=42); val=d.iloc[vi]
pack=joblib.load(R/'mejora_taxis/v2/referencia_81_validacion.joblib')
x=create_features(val).to_numpy(dtype='float32')
medians=pack['imputer'].statistics_[:81].astype('float32')
x=np.where(np.isfinite(x),x,medians).astype('float32')
base=pack['model'].predict(x)
score=rmsle(val.trip_duration,np.expm1(np.maximum(base,0)))
print('Referencia restaurada',score,flush=True)
np.save(O/'referencia_zvalid.npy',base)
assert abs(score-.37579821168828)<.00005
hist=pd.read_csv(O/'experimentos_wavelet.csv').to_dict('records')
hist.append({'modelo':'referencia_81_anterior','rmsle_validacion':score,'best_iteration':0,'n_variables':81})
for name in ['Fourier_Haar_fijas','Fourier_demanda_Haar','Fourier_demanda_target_OOF']:
    pred=np.load(O/(name+'_zvalid.npy'))
    for w in [.25,.5,.75]:
        z=w*base+(1-w)*pred
        hist.append({'modelo':f'blend_ref_{w:g}_con_{name}','rmsle_validacion':rmsle(val.trip_duration,np.expm1(np.maximum(z,0))),'best_iteration':0,'n_variables':0})
pd.DataFrame(hist).sort_values('rmsle_validacion').to_csv(O/'experimentos_wavelet.csv',index=False)
print(pd.DataFrame(hist).sort_values('rmsle_validacion').head(13).to_string(index=False))
