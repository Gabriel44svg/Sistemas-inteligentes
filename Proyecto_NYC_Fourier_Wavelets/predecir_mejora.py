"""Reproduce la entrega usando exclusivamente datos de test.csv y un modelo entrenado."""
from pathlib import Path
import joblib,numpy as np,pandas as pd
from features_taxi import create_features
# Es importante importar el módulo que define Augment para cargar el modelo.
from _mejora_geografia_v2 import Augment
from entrenar_mejora_final import GEO

def predecir(csv='test.csv',output='submission_reproducida.csv',modelo='mejora_taxis/modelo_compacto.joblib'):
    root=Path(__file__).parent
    raw=pd.read_csv(root/csv,parse_dates=['pickup_datetime'])
    pack=joblib.load(root/modelo)
    geo=pack['augment'].transform(raw).reset_index(drop=True)
    merged=create_features(raw).reset_index(drop=True)
    merged=pd.concat([merged,geo[GEO]],axis=1)
    for key in pack['encoder_maps']['keys']:
        idx=geo[key].to_numpy(dtype=np.int32)
        values=pack['encoder_maps']['means'][key]
        merged['oof_mean_'+key]=values['mean'][idx]
        merged['log_n_'+key]=values['log_count'][idx]
    names=pack['columnas_todas']
    x=pd.DataFrame(pack['imputer'].transform(merged[names]).astype('float32'),columns=names)
    logs=np.zeros(len(raw),dtype=np.float64)
    for component in pack['modelos']:
        logs+=component['peso']*component['modelo'].predict(x[component['columnas']])
    result=pd.DataFrame({'id':raw.id,'trip_duration':np.expm1(np.maximum(logs,0))})
    assert len(result)==len(raw) and result.id.is_unique and result.trip_duration.ge(0).all()
    result.to_csv(root/output,index=False)
    print('Archivo:',root/output,'filas:',len(result))
    return result

if __name__=='__main__':predecir()
