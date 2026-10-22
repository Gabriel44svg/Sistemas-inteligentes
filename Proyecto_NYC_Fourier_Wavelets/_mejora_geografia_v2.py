import os,sys,time,json,traceback
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import pandas as pd,numpy as np, joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from lightgbm import LGBMRegressor,early_stopping,log_evaluation
from sklearn.metrics import mean_squared_log_error,mean_absolute_error
from features_taxi import create_features
R=Path(__file__).resolve().parent; out=R/'mejora_taxis'/'v2';out.mkdir(parents=True,exist_ok=True)
t0=time.time()
def log(*xs): print(round(time.time()-t0,1),'s',*xs,flush=True)
def rmsle(y,p):return float(np.sqrt(mean_squared_log_error(y,np.maximum(p,0))))
def latlon_to_xy(lat,lon):return np.column_stack(((lon+73.99)*84.0,(lat-40.75)*111.2)).astype(np.float32)
def cluster_xy(fr,col):
    a=latlon_to_xy(fr[f'{col}_latitude'],fr[f'{col}_longitude']);return np.nan_to_num(a,nan=-100.,posinf=-100.,neginf=-100.)

class Augment:
 def __init__(self,n_clusters=90):self.n_clusters=n_clusters
 def fit(self,fr):
    both=np.concatenate([cluster_xy(fr,'pickup'),cluster_xy(fr,'dropoff')],axis=0)
    rng=np.random.default_rng(42);ix=rng.choice(len(both),min(175000,len(both)),replace=False)
    self.cluster=MiniBatchKMeans(n_clusters=self.n_clusters,n_init=2,batch_size=8192,max_iter=125,random_state=42).fit(both[ix])
    return self
 def transform(self,fr):
    op=cluster_xy(fr,'pickup');de=cluster_xy(fr,'dropoff');a=pd.DataFrame(index=fr.index)
    a['pickup_cluster']=self.cluster.predict(op).astype(np.int16)
    a['dropoff_cluster']=self.cluster.predict(de).astype(np.int16)
    a['cluster_pair']=(a.pickup_cluster.astype('int32')*self.n_clusters+a.dropoff_cluster).astype('int32')
    # Natural path grid with 200-400m cells.
    for prefix,xy in [('pickup',op),('dropoff',de)]:
        a[f'{prefix}_x']=xy[:,0]
        a[f'{prefix}_y']=xy[:,1]
        a[f'{prefix}_axis']=xy[:,1]-xy[:,0]*.25
        # Manhattan: in NYC 29 degree rotated grid coordinate; geometry deterministic.
        a[f'{prefix}_manh_a']=xy[:,0]*.87+xy[:,1]*.49
        a[f'{prefix}_manh_b']=-xy[:,0]*.49+xy[:,1]*.87
        gx=np.clip(np.floor((xy[:,0]+35)/.45).astype(np.int32),0,999)
        gy=np.clip(np.floor((xy[:,1]+40)/.45).astype(np.int32),0,999)
        a[f'{prefix}_grid_code']=(gx*1000+gy).astype(np.int32)
    dt=pd.to_datetime(fr.pickup_datetime)
    a['hour_bucket']=(dt.dt.hour//3).astype(np.int8)
    a['weekday_bucket']=(dt.dt.dayofweek).astype(np.int8)
    a['cluster_hour']=(a.pickup_cluster*8+a.hour_bucket).astype(np.int32)
    a['pickup_day_hour']=(a.pickup_cluster*56+a.weekday_bucket*8+a.hour_bucket).astype(np.int32)
    a['route_hour']=(a.cluster_pair*8+a.hour_bucket).astype(np.int32)
    a['od_manhattan_axis_delta']=a.dropoff_manh_a-a.pickup_manh_a
    a['od_manhattan_cross_delta']=a.dropoff_manh_b-a.pickup_manh_b
    a['od_manhattan_axis_abs']=np.abs(a.od_manhattan_axis_delta)
    a['od_manhattan_cross_abs']=np.abs(a.od_manhattan_cross_delta)
    # Complements for route/varying congestion.
    a['manh_axis_distance']=np.abs(a.od_manhattan_axis_delta)+np.abs(a.od_manhattan_cross_delta)
    return a

class LeakageFreeEncoder:
 def __init__(self,keys=['cluster_pair','cluster_hour','pickup_day_hour','route_hour'], smooth=65, folds=5):self.keys=keys;self.smooth=smooth;self.folds=folds
 def _map(self,train_f,target,out_f,key):
    df=pd.DataFrame({'k':train_f[key].values,'y':target})
    st=df.groupby('k',sort=False).y.agg(['mean','count'])
    mu=np.mean(target)
    values=(st['mean']*st['count']+mu*self.smooth)/(st['count']+self.smooth)
    pred=out_f[key].map(values).fillna(mu).to_numpy(dtype=np.float32)
    cnt=out_f[key].map(st['count']).fillna(0).to_numpy(dtype=np.float32)
    return pred,np.log1p(cnt)
 def fit_transform(self,fr,logtarget):
    results={k:np.empty(len(fr),dtype=np.float32) for k in self.keys}
    counts={k:np.empty(len(fr),dtype=np.float32) for k in self.keys}
    kfold=KFold(n_splits=self.folds,shuffle=True,random_state=47)
    for it,iv in kfold.split(fr):
        for k in self.keys:
            results[k][iv],counts[k][iv]=self._map(fr.iloc[it],logtarget[it],fr.iloc[iv],k)
    return pd.DataFrame({**{f'oof_mean_{k}':v for k,v in results.items()},**{f'log_n_{k}':v for k,v in counts.items()}},index=fr.index)
 def fit(self,fr,logtarget):self.ref=fr.reset_index(drop=True);self.logtarget=np.asarray(logtarget);return self
 def transform(self,fr):
    res={}
    for k in self.keys:
        preds,counts=self._map(self.ref,self.logtarget,fr,k)
        res[f'oof_mean_{k}']=preds;res[f'log_n_{k}']=counts
    return pd.DataFrame(res,index=fr.index)

def run():
 log('Cargando datos')
 df=pd.read_csv(R/'train.csv',parse_dates=['pickup_datetime']); idx=np.arange(len(df))
 itr,itemp=train_test_split(idx,test_size=.20,random_state=42)
 iva,iho=train_test_split(itemp,test_size=.50,random_state=42)
 tr=df.iloc[itr].copy();va=df.iloc[iva].copy();ho=df.iloc[iho].copy()
 tr=tr.loc[tr.trip_duration.between(1,43200)].copy()
 ytr=tr.trip_duration.to_numpy(dtype=np.float64);ztr=np.log1p(ytr)
 yva=va.trip_duration.to_numpy(dtype=np.float64);zva=np.log1p(yva)
 log('Partición',len(tr),len(va),len(ho))
 aug=Augment(90).fit(tr)
 feats={}
 for name,part in [('train',tr),('valid',va),('holdout',ho)]:
    log('Creando características',name)
    base=create_features(part).reset_index(drop=True)
    geo=aug.transform(part).reset_index(drop=True)
    feats[name]=pd.concat([base,geo],axis=1)
 gtrain=feats['train'];gvalid=feats['valid'];ghold=feats['holdout']
 catcols=['pickup_cluster','dropoff_cluster','cluster_pair','pickup_grid_code','dropoff_grid_code','cluster_hour','pickup_day_hour','route_hour']
 log('Codificando sin fuga: OOF entrenamiento, mapa completo solo validación')
 enc=LeakageFreeEncoder(smooth=80,folds=5)
 enc_train=enc.fit_transform(gtrain,ztr)
 enc.fit(gtrain,ztr)
 enc_valid=enc.transform(gvalid)
 enc_hold=enc.transform(ghold)
 Xtr=pd.concat([gtrain,enc_train],axis=1)
 Xva=pd.concat([gvalid,enc_valid],axis=1)
 Xho=pd.concat([ghold,enc_hold],axis=1)
 Xcols=list(Xtr.columns)
 imp=SimpleImputer(strategy='median',keep_empty_features=True)
 Xtr=pd.DataFrame(imp.fit_transform(Xtr).astype(np.float32),columns=Xcols)
 Xva=pd.DataFrame(imp.transform(Xva[Xcols]).astype(np.float32),columns=Xcols)
 Xho=pd.DataFrame(imp.transform(Xho[Xcols]).astype(np.float32),columns=Xcols)
 # Conservamos de las nuevas variables solamente las continuas y geométricas,
 # evitando códigos de categorías muy específicos que sobreajustaron el primer intento.
 cols_geo=['pickup_x','pickup_y','pickup_axis','pickup_manh_a','pickup_manh_b',
           'dropoff_x','dropoff_y','dropoff_axis','dropoff_manh_a','dropoff_manh_b',
           'od_manhattan_axis_delta','od_manhattan_cross_delta',
           'od_manhattan_axis_abs','od_manhattan_cross_abs','manh_axis_distance']
 orig_cols=list(create_features(tr.iloc[:1]).columns)
 params={'n_estimators':1600,'learning_rate':.045,'num_leaves':105,
         'max_depth':-1,'min_child_samples':75,'colsample_bytree':.92,
         'subsample':.9,'subsample_freq':1,'reg_lambda':5}
 opts=[('referencia_81',False,params),('rotacion_urbana',False,params),
       ('rotacion_mas_oof',True,params)]

 hist=[];zpreds={}
 for name,uses_oof,params in opts:
    if name=='referencia_81': cols=orig_cols
    elif not uses_oof: cols=orig_cols+cols_geo
    else: cols=orig_cols+cols_geo+[c for c in Xcols if c.startswith(('oof_','log_n_'))]
    log('Entrenando',name,'columnas',len(cols))
    m=LGBMRegressor(**params,objective='regression',random_state=42,n_jobs=4,verbosity=-1)
    m.fit(Xtr[cols],ztr,eval_set=[(Xva[cols],zva)],eval_metric='rmse',callbacks=[early_stopping(100,verbose=False)],categorical_feature=[])
    p=m.predict(Xva[cols]);real=np.expm1(np.maximum(p,0))
    score=rmsle(yva,real)
    hist.append({'modelo':name,'rmsle_validacion':score,'mejor_iteracion':int(m.best_iteration_),'n_features':len(cols),'segundos':round(time.time()-t0,1)})
    zpreds[name]=p
    log(name,'RMSLE',score,'iter',m.best_iteration_)
    pd.DataFrame(hist).to_csv(out/'experimentos.csv',index=False)
    if name=='rotacion_mas_oof':
        log('guardando mapa de importancias')
        pd.DataFrame({'feature':cols,'gain':m.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False).to_csv(out/'importancia_geo_oof.csv',index=False)
    if True:
        joblib.dump({'model':m,'cols':cols,'augment':aug,'encoder':enc,'imputer':imp},out/(name+'_validacion.joblib'),compress=2)
 log('Validando combinaciones en escala logarítmica')
 for base,other in [('referencia_81','rotacion_urbana'),('referencia_81','rotacion_mas_oof')]:
    for w in [0.25,0.5,0.75]:
        zz=w*zpreds[base]+(1-w)*zpreds[other]
        sc=rmsle(yva,np.expm1(np.maximum(0,zz)))
        hist.append({'modelo':f'blend_{base}_{other}_{w}','rmsle_validacion':sc,'mejor_iteracion':0,'n_features':len(Xcols)})
 pd.DataFrame(hist).sort_values('rmsle_validacion').to_csv(out/'experimentos.csv',index=False)
 log('Ranking',sorted([(r['modelo'],round(r['rmsle_validacion'],6)) for r in hist],key=lambda x:x[1]))
 # Holdout only winner once; do not use it for tuning.
 top=sorted(hist,key=lambda x:x['rmsle_validacion'])[0]
 name=top['modelo'];log('Ganador',name)
 if name.startswith('blend_'):
    assert '_rotacion_' in name
    w=float(name.rsplit('_',1)[-1])
    other='rotacion_urbana' if 'rotacion_urbana' in name else 'rotacion_mas_oof'
    comps=[('referencia_81',w),(other,1-w)]
 else: comps=[(name,1)]
 pp=0
 for nn,w in comps:
    pack=joblib.load(out/(nn+'_validacion.joblib'))
    if pack is None: raise RuntimeError('Ganador inesperado sin modelo serializado')
    pp+=w*pack['model'].predict(Xho[pack['cols']])
 ppho=np.expm1(np.maximum(0,pp));score_ho=rmsle(ho.trip_duration,ppho)
 log('Holdout (una sola medición)',score_ho)
 json.dump({'ganador':top,'holdout_rmsle':score_ho,'resultados':hist,'train_n':len(tr),'valid_n':len(va),'holdout_n':len(ho),'semilla':42},open(out/'metricas_mejora.json','w'),indent=2)
 log('EXPERIMENTO COMPLETO')
if __name__=='__main__':
 try:run()
 except Exception as e:log('ERROR',str(e));traceback.print_exc();sys.exit(1)
