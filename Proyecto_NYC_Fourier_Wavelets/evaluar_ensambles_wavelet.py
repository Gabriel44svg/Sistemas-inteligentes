"""Compara Fourier/Haar con el mejor ensamble anterior, exclusivamente en validacion."""
from pathlib import Path
import pandas as pd,numpy as np,joblib,json,gc
from sklearn.model_selection import train_test_split
from features_taxi import create_features
from wavelet_fft_taxi import rmsle
import __main__
from _mejora_geografia_v2 import Augment,LeakageFreeEncoder
__main__.Augment=Augment;__main__.LeakageFreeEncoder=LeakageFreeEncoder
R=Path(__file__).resolve().parent;O=R/'resultados_wavelet'

def run():
    df=pd.read_csv(R/'train.csv',parse_dates=['pickup_datetime'])
    ids=np.arange(len(df));_, temp=train_test_split(ids,test_size=.2,random_state=42)
    vi,_=train_test_split(temp,test_size=.5,random_state=42)
    val=df.iloc[vi].reset_index(drop=True);y=val.trip_duration.to_numpy()
    base=np.load(O/'referencia_zvalid.npy')
    pack=joblib.load(R/'mejora_taxis/v2/rotacion_mas_oof_validacion.joblib')
    raw=pd.concat([create_features(val),pack['augment'].transform(val)],axis=1)
    aug=pack['encoder'].transform(raw)
    xfull=pd.concat([raw,aug],axis=1)
    ordered=list(pack['imputer'].feature_names_in_)
    X=pack['imputer'].transform(xfull[ordered])
    colidx=[ordered.index(c) for c in pack['cols']]
    a=pack['model'].predict(X[:,colidx])
    previous=(base+a)/2.
    baseline=rmsle(y,np.expm1(np.maximum(previous,0)))
    print('Validación ensamble anterior reconstruida:',baseline,flush=True)
    assert abs(baseline-.374624017)<.00004,'Particion o calculo no coincide con referencia'
    np.save(O/'ensamble_previo_zvalid.npy',previous)
    del pack,raw,aug,xfull,X,df;gc.collect()
    old=pd.read_csv(O/'experimentos_wavelet.csv').to_dict('records')
    for name in ['Fourier_Haar_fijas','Fourier_demanda_Haar','Fourier_demanda_target_OOF']:
        z=np.load(O/(name+'_zvalid.npy'))
        for w_old in [.25,.5,.75,.9]:
            p=w_old*previous+(1-w_old)*z
            old.append({'modelo':f'ensemble_previo_{w_old:g}_con_{name}',
                        'rmsle_validacion':rmsle(y,np.expm1(np.maximum(p,0))),
                        'best_iteration':0,'n_variables':0})
    old.append({'modelo':'ensamble_previo_0.5_0.5','rmsle_validacion':baseline,'best_iteration':0,'n_variables':0})
    ranking=pd.DataFrame(old).sort_values('rmsle_validacion').reset_index(drop=True)
    ranking.to_csv(O/'comparacion_completa.csv',index=False)
    winner=ranking.iloc[0].to_dict()
    with open(O/'ganador_final_validacion.json','w') as f:
        json.dump({'mejor':winner,'rmsle_antes':baseline,'metrica':'RMSLE',
                   'regla':'Seleccion exclusiva con validacion; holdout no consultado'},f,indent=2)
    print('TOP VAL:',ranking[['modelo','rmsle_validacion']].head(15).to_string(index=False),flush=True)
if __name__=='__main__':run()
