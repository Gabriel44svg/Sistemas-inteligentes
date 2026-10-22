"""Experimento reproducible: misma particion que el trabajo anterior; holdout sellado.
python experimento_wavelets_fft.py
"""
import os,gc,time,json,sys
os.environ.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2'})
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor,early_stopping
from features_taxi import create_features
from wavelet_fft_taxi import deterministic_features,FourierHaarEncoder,rmsle
# Compatibilidad con joblib: modelos anteriores se serializaron con clases de __main__.
import __main__
from _mejora_geografia_v2 import Augment,LeakageFreeEncoder
__main__.Augment=Augment;__main__.LeakageFreeEncoder=LeakageFreeEncoder

R=Path(__file__).resolve().parent; O=R/'resultados_wavelet';O.mkdir(exist_ok=True)
t0=time.time()
def say(*a):print('%.1fs'%(time.time()-t0),*a,flush=True)

def run():
    d=pd.read_csv(R/'train.csv',parse_dates=['pickup_datetime']);say('loaded',d.shape)
    it,temp=train_test_split(np.arange(len(d)),test_size=.2,random_state=42)
    iv,ih=train_test_split(temp,test_size=.5,random_state=42)
    tr=d.iloc[it].copy();va=d.iloc[iv].copy()
    tr=tr[tr.trip_duration.between(1,43200)].copy()
    z=np.log1p(tr.trip_duration.to_numpy(dtype='float64'))
    zv=np.log1p(va.trip_duration.to_numpy(dtype='float64'))
    say('train valid holdout',len(tr),len(va),len(ih),'extremos validacion',int((va.trip_duration>43200).sum()))
    a=create_features(tr).reset_index(drop=True); b=create_features(va).reset_index(drop=True)
    extra_tr=deterministic_features(tr).reset_index(drop=True)
    extra_va=deterministic_features(va).reset_index(drop=True)
    say('fourier/haar bases',extra_tr.shape[1]);
    enc=FourierHaarEncoder(folds=5,seed=47)
    wave_tr=enc.fit_transform(tr,z,supervised=True).reset_index(drop=True)
    enc.fit(tr,z,supervised=True)
    wave_va=enc.transform(va).reset_index(drop=True)
    say('demanda Haar FFT + OOF etiquetas espectrales',wave_tr.shape)
    # Imputacion SOLO train, con numpy para limitar memoria.
    Xtr=pd.concat([a,extra_tr,wave_tr],axis=1).astype('float32')
    Xva=pd.concat([b,extra_va,wave_va],axis=1).astype('float32')
    del a,b,extra_tr,extra_va,wave_tr,wave_va;gc.collect()
    cols=list(Xtr.columns)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    vtr=imp.fit_transform(Xtr).astype('float32');vva=imp.transform(Xva).astype('float32')
    del Xtr,Xva;gc.collect()
    say('matrices',vtr.shape,vva.shape)
    c0=81;cd=c0+len(deterministic_features(tr.iloc[:2]).columns)
    subsets={'Fourier_Haar_fijas':np.arange(cd),
             'Fourier_demanda_Haar':np.arange(len(cols)-14),
             'Fourier_demanda_target_OOF':np.arange(len(cols))}
    for name,ix in subsets.items():
        if name=='Fourier_Haar_fijas':
            # primeras 81 originales + deterministas
            pass
    params=dict(objective='regression',learning_rate=.045,n_estimators=1700,
       num_leaves=105,max_depth=-1,min_child_samples=75,colsample_bytree=.92,
       subsample=.90,subsample_freq=1,reg_lambda=5,random_state=42,n_jobs=4,verbosity=-1)
    results=[]; preds={}
    for name,ix in subsets.items():
        say('training',name,'nfeatures',len(ix))
        model=LGBMRegressor(**params)
        model.fit(vtr[:,ix],z,eval_set=[(vva[:,ix],zv)],eval_metric='rmse',callbacks=[early_stopping(110,verbose=False)])
        pv=model.predict(vva[:,ix]); sc=rmsle(va.trip_duration.to_numpy(),np.expm1(np.maximum(pv,0)))
        preds[name]=pv.astype('float32')
        row={'modelo':name,'rmsle_validacion':sc,'best_iteration':model.best_iteration_,'n_variables':len(ix),'segundos':round(time.time()-t0,1)}
        say('RESULT',row)
        results.append(row)
        pd.DataFrame(results).to_csv(O/'experimentos_wavelet.csv',index=False)
        np.save(O/(name+'_zvalid.npy'),pv)
        if name=='Fourier_demanda_target_OOF':
            joblib.dump({'modelo':model,'columnas':cols,'seleccion':ix,'imputer':imp,'encoder':enc,'base':'features_taxi.create_features','version':'1'},O/'wavelet_modelo_validacion.joblib',compress=3)
            gains=model.booster_.feature_importance(importance_type='gain')
            pd.DataFrame({'variable':np.array(cols)[ix],'gain':gains}).sort_values('gain',ascending=False).to_csv(O/'importancia_wavelet.csv',index=False)
        del model;gc.collect()
    say('obteniendo predicciones del modelo base preexistente para blends')
    pack=joblib.load(R/'mejora_taxis/v2/referencia_81_validacion.joblib')
    # el imputador original fue ajustado con features base y geométricas; primeras 81 = mismas features
    simple=create_features(va).to_numpy(dtype='float32')
    stats=pack['imputer'].statistics_[:81].astype('float32')
    np.copyto(simple,np.where(np.isfinite(simple),simple,stats))
    base=pack['model'].predict(simple).astype('float32')
    del pack,simple;gc.collect()
    np.save(O/'referencia_zvalid.npy',base)
    yva=va.trip_duration.to_numpy()
    ss=rmsle(yva,np.expm1(np.maximum(base,0)))
    results.append({'modelo':'referencia_81_anterior','rmsle_validacion':ss,'best_iteration':0,'n_variables':81})
    for name,p in preds.items():
        for w in (.25,.5,.75):
            zz=w*base+(1-w)*p
            score=rmsle(yva,np.expm1(np.maximum(zz,0)))
            results.append({'modelo':f'blend_ref_{w:g}_con_{name}','rmsle_validacion':score,'best_iteration':0,'n_variables':0})
    pd.DataFrame(results).sort_values('rmsle_validacion').to_csv(O/'experimentos_wavelet.csv',index=False)
    top=sorted(results,key=lambda r:r['rmsle_validacion'])[0]
    say('GANADOR VALIDACION',top)
    with open(O/'validacion.json','w') as fo:
        json.dump({'ganador':top,'antes_mejor_ensamble_validacion':.3746240172258088,'base_validacion':.37579821168827976,
                   'holdout_reservado_sin_evaluar':True,'n_train':len(tr),'n_val':len(va),'n_holdout':len(ih)},fo,indent=2)
    del vtr,vva,d;gc.collect()
    say('finished, no holdout consulted')

if __name__=='__main__':
    try:run()
    except Exception as e:
        import traceback
        traceback.print_exc();sys.exit(1)
