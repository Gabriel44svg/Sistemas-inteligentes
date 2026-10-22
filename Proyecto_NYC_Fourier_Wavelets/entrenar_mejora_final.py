"""Reentrenamiento final de una mejora MODESTA, sin leakage: OOF mediante bincount."""
import json,time,gc
from pathlib import Path
import numpy as np,pandas as pd,joblib
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from features_taxi import create_features
from _mejora_geografia_v2 import Augment
R=Path(__file__).parent;O=R/'mejora_taxis';O.mkdir(exist_ok=True)
KEYS=['cluster_pair','cluster_hour','pickup_day_hour','route_hour'];SMOOTH=80.
GEO=['pickup_x','pickup_y','pickup_axis','pickup_manh_a','pickup_manh_b','dropoff_x','dropoff_y','dropoff_axis','dropoff_manh_a','dropoff_manh_b','od_manhattan_axis_delta','od_manhattan_cross_delta','od_manhattan_axis_abs','od_manhattan_cross_abs','manh_axis_distance']

def encode_oof_and_maps(train_keys,z,test_keys):
    n=len(z);kf=KFold(n_splits=5,shuffle=True,random_state=47)
    result={};ptest={};pack={};mu_all=z.mean()
    for name in KEYS:
        values=train_keys[name].to_numpy(dtype=np.int32,copy=True)
        unseen=test_keys[name].to_numpy(dtype=np.int32,copy=True)
        size=int(max(values.max(),unseen.max()))+1
        oof=np.zeros(n,dtype=np.float32);ocnt=np.zeros(n,dtype=np.float32)
        for tr,va in kf.split(np.arange(n)):
            counts=np.bincount(values[tr],minlength=size)
            sums=np.bincount(values[tr],weights=z[tr],minlength=size)
            mu=z[tr].mean()
            means=(sums+mu*SMOOTH)/(counts+SMOOTH)
            oof[va]=means[values[va]].astype(np.float32)
            ocnt[va]=np.log1p(counts[values[va]]).astype(np.float32)
        counts=np.bincount(values,minlength=size)
        sums=np.bincount(values,weights=z,minlength=size)
        mean_full=((sums+mu_all*SMOOTH)/(counts+SMOOTH)).astype(np.float32)
        cnt_full=np.log1p(counts).astype(np.float32)
        result['oof_mean_'+name]=oof;result['log_n_'+name]=ocnt
        ptest['oof_mean_'+name]=mean_full[unseen]
        ptest['log_n_'+name]=cnt_full[unseen]
        pack[name]={'mean':mean_full,'log_count':cnt_full}
        print('  OOF',name,'size',size,flush=True)
    return pd.DataFrame(result),pd.DataFrame(ptest),{'means':pack,'keys':KEYS}

def entrenar_final():
    t=time.time();print('Cargando train/test',flush=True)
    full=pd.read_csv(R/'train.csv',parse_dates=['pickup_datetime'])
    test=pd.read_csv(R/'test.csv',parse_dates=['pickup_datetime'])
    assert full.id.is_unique and test.id.is_unique and len(test)==100000
    full=full.loc[full.trip_duration.between(1,43200)].copy()
    z=np.log1p(full.trip_duration.to_numpy(dtype=np.float64))
    aug=Augment(90).fit(full)
    bgfull=create_features(full).reset_index(drop=True)
    bgtest=create_features(test).reset_index(drop=True)
    geo_full=aug.transform(full).reset_index(drop=True)
    geo_test=aug.transform(test).reset_index(drop=True)
    print('Ingeniería terminada',round(time.time()-t,1),'s',flush=True)
    oof,ptest,compact_maps=encode_oof_and_maps(geo_full[KEYS],z,geo_test[KEYS])
    train_matrix=pd.concat([bgfull,geo_full[GEO],oof],axis=1)
    test_matrix=pd.concat([bgtest,geo_test[GEO],ptest],axis=1)
    columns=list(train_matrix.columns);orig_cols=list(bgfull.columns)
    del bgfull,bgtest,geo_full,geo_test,oof,ptest,full;gc.collect()
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    arr=imp.fit_transform(train_matrix).astype(np.float32,copy=False)
    arr_test=imp.transform(test_matrix[columns]).astype(np.float32,copy=False)
    del train_matrix,test_matrix;gc.collect()
    train_matrix=pd.DataFrame(arr,columns=columns,copy=False)
    test_matrix=pd.DataFrame(arr_test,columns=columns,copy=False)
    orig=orig_cols
    xlist=[('referencia_81',.5,orig),('rotacion_mas_oof',.5,columns)]
    params={'n_estimators':1600,'learning_rate':.045,'num_leaves':105,'max_depth':-1,'min_child_samples':75,'colsample_bytree':.92,'subsample':.9,'subsample_freq':1,'reg_lambda':5,'objective':'regression','random_state':42,'n_jobs':4,'verbosity':-1}
    log_pred=np.zeros(len(test),dtype=np.float64)
    models=[]
    for name,w,cols in xlist:
        print('Entrenando',name,'features',len(cols),'s',round(time.time()-t,1),flush=True)
        model=LGBMRegressor(**params).fit(train_matrix[cols],z)
        log_pred+=w*model.predict(test_matrix[cols])
        model.booster_.save_model(str(O/(name+'_final.txt')))
        models.append({'nombre':name,'peso':w,'columnas':cols,'modelo':model})
        gc.collect()
    preds=np.expm1(np.maximum(log_pred,0))
    submission=pd.DataFrame({'id':test.id.to_numpy(),'trip_duration':preds})
    assert submission.columns.tolist()==['id','trip_duration']
    assert len(submission)==len(test)==100000
    assert submission.id.is_unique and (submission.trip_duration>=0).all() and np.isfinite(submission.trip_duration).all()
    subfile=O/'submission_experimental.csv';submission.to_csv(subfile,index=False)
    assert np.array_equal(pd.read_csv(subfile)['id'].to_numpy(),test['id'].to_numpy())
    pack={'modelos':models,'imputer':imp,'augment':aug,'encoder_maps':compact_maps,'columnas_todas':columns,'columnas_geo':GEO}
    joblib.dump(pack,O/'modelo_compacto.joblib',compress=3)
    meta={'filas_entrenadas':len(z),'test_filas':len(test),'pesos':{m['nombre']:m['peso'] for m in models},'rmsle_validacion':0.3746240172258088,'rmsle_holdout':0.37928817537192905,'nota':'La métrica de test.csv ciego no se conoce. No está garantizada mejora sobre 0.3704 externo.','duracion_s':round(time.time()-t,1)}
    (O/'meta_final.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    print('Listo',subfile, 'MIN/MAX/MED',float(preds.min()),float(preds.max()),float(np.median(preds)),'s',round(time.time()-t,1),flush=True)
if __name__=='__main__':entrenar_final()
