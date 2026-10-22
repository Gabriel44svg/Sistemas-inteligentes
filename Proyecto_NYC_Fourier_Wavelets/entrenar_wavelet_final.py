"""Entrenamiento final Fourier/Haar (solo despues de eleccion con validacion).

Se reentrena componente espectral en train completo elegible y se combina con
el ensamble previo 75/25 en escala log. No se usan etiquetas test ni holdout.
"""
import os,time,json,gc
os.environ.update({'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'2'})
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from features_taxi import create_features
from wavelet_fft_taxi import deterministic_features,FourierHaarEncoder
R=Path(__file__).resolve().parent;O=R/'resultados_wavelet';O.mkdir(exist_ok=True)
t0=time.time()
def say(*a):print('%.1fs'%(time.time()-t0),*a,flush=True)

def main():
    full=pd.read_csv(R/'train.csv',parse_dates=['pickup_datetime'])
    test=pd.read_csv(R/'test.csv',parse_dates=['pickup_datetime'])
    assert len(test)==100000 and full.id.is_unique and test.id.is_unique
    tr=full.loc[full.trip_duration.between(1,43200)].copy();del full;gc.collect()
    z=np.log1p(tr.trip_duration.to_numpy(dtype='float64'))
    say('Entrenar %d filas | predecir %d'%(len(tr),len(test)))
    # OOF: cada muestra de train obtiene señales aprendidas sin su propia etiqueta.
    enc=FourierHaarEncoder(folds=5,seed=47)
    enc_tr=enc.fit_transform(tr,z,supervised=True).reset_index(drop=True)
    enc.fit(tr,z,supervised=True)
    enc_te=enc.transform(test).reset_index(drop=True)
    say('Wavelets de demanda y estadisticas OOF listas')
    Xtr=pd.concat([create_features(tr).reset_index(drop=True),
                   deterministic_features(tr).reset_index(drop=True),enc_tr],axis=1)
    Xte=pd.concat([create_features(test).reset_index(drop=True),
                   deterministic_features(test).reset_index(drop=True),enc_te],axis=1)
    del tr,enc_tr,enc_te;gc.collect()
    cols=list(Xtr.columns)
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    a=imp.fit_transform(Xtr).astype('float32');b=imp.transform(Xte[cols]).astype('float32')
    del Xtr,Xte;gc.collect();say('Matriz final',a.shape,b.shape)
    params={'objective':'regression','learning_rate':.045,'n_estimators':1700,
       'num_leaves':105,'max_depth':-1,'min_child_samples':75,
       'colsample_bytree':.92,'subsample':.90,'subsample_freq':1,
       'reg_lambda':5,'random_state':42,'n_jobs':4,'verbosity':-1}
    model=LGBMRegressor(**params)
    say('Inicia LightGBM Fourier + Haar')
    model.fit(a,z)
    zspec=model.predict(b);del a,b;gc.collect()
    say('Predicciones espectrales completadas')
    old=pd.read_csv(R/'mejora_taxis'/'submission_experimental.csv')
    assert old.columns.tolist()==['id','trip_duration'] and old.id.equals(test.id)
    assert old.trip_duration.ge(0).all()
    zold=np.log1p(old.trip_duration.to_numpy(dtype='float64'))
    znew=.75*zold+.25*zspec
    preds=np.expm1(np.maximum(znew,0))
    result=pd.DataFrame({'id':test.id.to_numpy(),'trip_duration':preds})
    assert len(result)==100000 and result.id.is_unique and np.isfinite(preds).all() and (preds>=0).all()
    path=R/'submission_wavelet_fft.csv';result.to_csv(path,index=False)
    reread=pd.read_csv(path)
    assert reread.columns.tolist()==['id','trip_duration'] and np.array_equal(reread.id.to_numpy(),test.id.to_numpy())
    assert np.allclose(reread.trip_duration, preds,atol=1e-7,rtol=1e-12)
    pack={'modelo':model,'encoder':enc,'imputer':imp,'columnas':cols,
          'pesos':{'ensamble_geografico_anterior':.75,'wavelet_fft':.25},
          'metodo':'Fourier temporal+espacial, Haar demanda, FFT/Haar target OOF'}
    joblib.dump(pack,O/'modelo_wavelet_final.joblib',compress=3)
    model.booster_.save_model(str(O/'modelo_wavelet_final.txt'))
    valid=json.load(open(O/'ganador_final_validacion.json'))
    ho=json.load(open(O/'holdout_final.json'))
    info={'modelo':'0.75 ensamble anterior + 0.25 Fourier/Haar','entrenamiento_filtrado':len(z),
          'test_filas':len(test),'validacion_rmsle':valid['mejor']['rmsle_validacion'],
          'holdout_rmsle':ho['rmsle_holdout_wavelet'],
          'rmsle_anterior_validacion':valid['rmsle_antes'],
          'rmsle_anterior_holdout':ho['rmsle_holdout_ensamble_previo'],
          'pesos':pack['pesos'],'n_features_wavelet':len(cols),
          'objetivo_de_0_20_alcanzado':False,
          'nota':'Score 0.3704 de test ciego reportado por usuario no se puede contrastar localmente',
          'tiempo_entrenamiento_seg':round(time.time()-t0,1)}
    with open(O/'meta_final_wavelet.json','w') as f:json.dump(info,f,indent=2,ensure_ascii=False)
    say('CSV LISTO',str(path),'min',float(preds.min()),'mediana',float(np.median(preds)),'max',float(preds.max()))

if __name__=='__main__':main()
