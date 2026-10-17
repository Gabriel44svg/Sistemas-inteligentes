"""Ejemplo: python predecir.py test.csv salida.csv (desde el directorio del proyecto)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from features_taxi import create_features

def main(src='test.csv',dst='submission_recalculada.csv'):
    d=pd.read_csv(src,parse_dates=['pickup_datetime'])
    b=joblib.load(Path(__file__).resolve().parent/'modelos'/'modelo_final.joblib')
    x=b['imputer'].transform(create_features(d)[b['columnas']]).astype(np.float32)
    log_preds=np.zeros(len(d),dtype=np.float64)
    for name,weight in b['componentes']:
        log_preds+=weight*np.maximum(b['modelos'][name].predict(x),0)
    y=np.expm1(np.clip(log_preds,0,20))
    output=pd.DataFrame({'id':d.id,'trip_duration':y})
    assert output.id.is_unique and np.isfinite(output.trip_duration).all() and output.trip_duration.ge(0).all()
    output.to_csv(dst,index=False,float_format='%.5f')
    print(f'Guardadas {len(output)} predicciones en {dst}')
if __name__=='__main__':main(*(sys.argv[1:]))
