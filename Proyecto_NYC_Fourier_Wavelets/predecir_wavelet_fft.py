"""Reproduccion desde los modelos guardados, sin volver a entrenar ni ver etiquetas test.
Uso: python predecir_wavelet_fft.py
"""
import numpy as np,pandas as pd,joblib
from pathlib import Path
from features_taxi import create_features
from wavelet_fft_taxi import deterministic_features
from entrenar_mejora_final import GEO
from _mejora_geografia_v2 import Augment,LeakageFreeEncoder
R=Path(__file__).resolve().parent

def predecir(input_csv='test.csv',output_csv='submission_wavelet_reproducida.csv'):
    test=pd.read_csv(R/input_csv,parse_dates=['pickup_datetime'])
    old=joblib.load(R/'mejora_taxis/modelo_compacto.joblib')
    geo=old['augment'].transform(test).reset_index(drop=True)
    src=pd.concat([create_features(test).reset_index(drop=True),geo[GEO]],axis=1)
    for key in old['encoder_maps']['keys']:
        idx=geo[key].to_numpy(dtype=np.int32)
        maps=old['encoder_maps']['means'][key]
        src['oof_mean_'+key]=maps['mean'][idx]
        src['log_n_'+key]=maps['log_count'][idx]
    cols=old['columnas_todas']
    mx=pd.DataFrame(old['imputer'].transform(src[cols]).astype('float32'),columns=cols)
    zold=np.zeros(len(test),dtype=np.float64)
    for comp in old['modelos']:
        zold+=comp['peso']*comp['modelo'].predict(mx[comp['columnas']])
    # Ensamble anterior guardaba predicciones truncadas >=0 segundos.
    zold=np.maximum(zold,0.)
    spectral=joblib.load(R/'resultados_wavelet/modelo_wavelet_final.joblib')
    features=pd.concat([create_features(test).reset_index(drop=True),
                        deterministic_features(test).reset_index(drop=True),
                        spectral['encoder'].transform(test).reset_index(drop=True)],axis=1)
    sx=spectral['imputer'].transform(features[spectral['columnas']]).astype('float32')
    zwave=spectral['modelo'].predict(sx)
    pred=np.expm1(np.maximum(.75*zold+.25*zwave,0))
    ans=pd.DataFrame({'id':test.id.to_numpy(),'trip_duration':pred})
    assert ans.columns.tolist()==['id','trip_duration']
    assert ans.id.is_unique and np.isfinite(pred).all() and np.all(pred>=0)
    ans.to_csv(R/output_csv,index=False)
    truth=R/'submission_wavelet_fft.csv'
    if input_csv=='test.csv' and truth.exists():
        ref=pd.read_csv(truth)
        assert np.array_equal(ref.id.to_numpy(),ans.id.to_numpy())
        difference=np.max(np.abs(ref.trip_duration.to_numpy()-pred))
        print('Diferencia maxima en segundos frente a submission original:',difference)
        assert difference<1e-6
    print('Guardado',R/output_csv,'filas',len(ans))
    return ans

if __name__=='__main__':predecir()
