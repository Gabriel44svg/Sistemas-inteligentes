"""Analisis espectral y Haar multiresolucion, sin usar etiquetas de validacion/test.

Las entradas supervisadas se calculan OOF: el fold objetivo jamas se usa para estimar
sus coeficientes. Las entradas de demanda usan solamente pickups de folds de train.
Haar se implementa con NumPy para no depender de PyWavelets.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.maximum(p, 0)) - np.log1p(y)) ** 2)))


def _slots(dt):
    dt = pd.to_datetime(dt)
    origin = pd.Timestamp('2016-01-01')
    sec = (dt - origin).dt.total_seconds().to_numpy()
    return np.clip(np.floor(sec / 1800).astype(np.int32), 0, 48 * 184 - 1)


def _time_fields(d):
    dt = pd.to_datetime(d['pickup_datetime'])
    hod = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy()/60 + dt.dt.second.to_numpy()/3600
    dow = dt.dt.dayofweek.to_numpy()
    weekphase = dow + hod/24
    doy = dt.dt.dayofyear.to_numpy() - 1 + hod/24
    return hod, weekphase, doy, dow * 48 + (hod * 2).astype('int32')


def deterministic_features(d):
    """Ciclos trigonométricos (Fourier) y bases de Haar, sin fit ni uso de y."""
    hod, weekphase, doy, slotweek = _time_fields(d)
    out = {}
    # Fourier: series armónicas diarias, semanales y estacionales.
    for j in [2, 3, 4, 6, 8, 12]:
        out[f'fourier_dia_sin_{j}'] = np.sin(2*np.pi*j*hod/24).astype('float32')
        out[f'fourier_dia_cos_{j}'] = np.cos(2*np.pi*j*hod/24).astype('float32')
    for j in [1, 2, 3, 4, 6, 8, 12]:
        out[f'fourier_semana_sin_{j}'] = np.sin(2*np.pi*j*weekphase/7).astype('float32')
        out[f'fourier_semana_cos_{j}'] = np.cos(2*np.pi*j*weekphase/7).astype('float32')
    for j in [2, 3, 4]:
        out[f'fourier_semestre_sin_{j}'] = np.sin(2*np.pi*j*doy/366).astype('float32')
        out[f'fourier_semestre_cos_{j}'] = np.cos(2*np.pi*j*doy/366).astype('float32')
    # Funciones de Haar a niveles horarios distintos. Son funciones fijas, no estadísticas aprendidas.
    for block in [2, 4, 8, 16, 32]:
        within = slotweek % block
        out[f'haar_fase_semana_{block}'] = np.where(within < block//2, 1., -1.).astype('float32')
    # FFT de posición geográfica: bases espaciales sinusoidales a distintas escalas.
    # Coords km aproximadas (en la latitud de NYC).
    for prefix in ('pickup', 'dropoff'):
        x = (pd.to_numeric(d[f'{prefix}_longitude'],errors='coerce').to_numpy() + 74.0) * 84.0
        y = (pd.to_numeric(d[f'{prefix}_latitude'],errors='coerce').to_numpy() - 40.75) * 111.2
        for distance in [2., 6.]:
            out[f'{prefix}_fourier_x_cos_{int(distance)}'] = np.cos(2*np.pi*x/distance).astype('float32')
            out[f'{prefix}_fourier_y_cos_{int(distance)}'] = np.cos(2*np.pi*y/distance).astype('float32')
    return pd.DataFrame(out, index=d.index)


def haar_scales(x, levels=(2,4,8,16,32,64)):
    """Aproximaciones y detalles exactos de la DWT Haar reconstruida por nivel.
    Para una serie x y bloques disjuntos dyadicos, A_s es la media de su bloque.
    D_s = A_(s/2) - A_s es la reconstruccion de detalle Haar ortogonal.
    """
    n=len(x); pad=1 << int(np.ceil(np.log2(max(n,2))))
    v=np.pad(x,(0,pad-n),mode='wrap').astype('float64')
    out={}
    prev=v.copy()
    for s in levels:
        coarse=np.repeat(v.reshape(-1,s).mean(axis=1),s)
        out[f'haar_A{s}']=coarse[:n].astype('float32')
        out[f'haar_D{s}']=(prev[:n]-coarse[:n]).astype('float32')
        prev=coarse
    return out


def fft_bands(v, intervals=(12, 48, 336)):
    """Componentes de baja frecuencia calculados por truncamiento espectral, en bins 30min."""
    v=np.asarray(v,dtype='float64')
    n=len(v); fft=np.fft.rfft(v); freq=np.fft.rfftfreq(n,d=1)
    out={}
    low_previous=np.zeros(n)
    for hours in intervals:
        # periodos de al menos 'hours' (horas), bins=2*hours.
        f=np.where(freq <= 1.0/(hours*2),fft,0)
        low=np.fft.irfft(f,n=n).real
        out[f'fft_lenta_{hours}h']=low.astype('float32')
        out[f'fft_banda_{hours}h']=(low-low_previous).astype('float32')
        low_previous=low
    return out


def _regions(d):
    lon=d.pickup_longitude.to_numpy();lat=d.pickup_latitude.to_numpy()
    return {
      'general':np.ones(len(d),dtype=bool),
      'manhattan':((lon>-74.035)&(lon<-73.93)&(lat>40.695)&(lat<40.835)),
      'aeropuertos':(((lat>40.61)&(lat<40.69)&(lon>-73.84)&(lon<-73.74))|
                     ((lat>40.745)&(lat<40.795)&(lon>-73.90)&(lon<-73.84))),
    }


def _demand_build(reference, nslots=48*184):
    """Solo coordenadas/tiempo, sin usar duraciones ni datos de test/valid."""
    ix=_slots(reference.pickup_datetime); regions=_regions(reference)
    signal={}
    for name,mask in regions.items():
        counts=np.bincount(ix[mask],minlength=nslots).astype('float64')
        # normalizar como conteo por cada 100k ejemplos del grupo de referencia.
        counts*=100000/max(len(reference),1)
        # estabilizar discretos; log ayuda a las colas raras
        x=np.log1p(counts)
        h=haar_scales(x,levels=(2,4,8,16,32,64))
        ft=fft_bands(x,intervals=(12,48,168))
        for field in ('haar_A2','haar_A8','haar_A32','haar_D2','haar_D8','haar_D32'):
            signal[f'dem_{name}_{field}']=h[field]
        for field in ('fft_lenta_12h','fft_banda_48h','fft_lenta_168h'):
            signal[f'dem_{name}_{field}']=ft[field]
    return signal


def _supervised_build(ref, z):
    """Media suavizada de log-duracion por media hora de semana y su FFT/Haar."""
    _,_,_,slot=_time_fields(ref)
    nbin=336
    cnt=np.bincount(slot,minlength=nbin).astype('float64')
    sums=np.bincount(slot,weights=z,minlength=nbin)
    mean=(sums+45*np.mean(z))/(cnt+45)
    # Reflejo de periodo semanal exacto: se extiende ciclicamente a 512 puntos.
    # Coeficientes Fourier armonicos principales de la funcion log-tiempo media.
    centered=mean-mean.mean(); f=np.fft.rfft(centered); freqs=np.fft.rfftfreq(nbin)
    sig={}
    for k in [4,8,16,32,64]:
        out=np.fft.irfft(np.where((np.arange(len(f))<=k),f,0),n=nbin).real
        sig[f'logtarget_fft_{k}']=out.astype('float32')
    h=haar_scales(mean,levels=(2,4,8,16,32))
    for s in (2,4,8,16):
        sig[f'logtarget_haar_A{s}']=h[f'haar_A{s}']
        sig[f'logtarget_haar_D{s}']=h[f'haar_D{s}']
    sig['logtarget_semana_suavizada']=mean.astype('float32')
    return sig


class FourierHaarEncoder:
    def __init__(self,folds=5,seed=47):
        self.folds=folds;self.seed=seed
    def fit_transform(self,tr,z,supervised=False):
        n=len(tr); out={}
        for ixtr,ixval in KFold(self.folds,shuffle=True,random_state=self.seed).split(tr):
            part=tr.iloc[ixtr]; dest=tr.iloc[ixval]
            ts=_slots(dest.pickup_datetime)
            signals=_demand_build(part)
            if supervised:
                _,_,_,weeks=_time_fields(dest)
                signals.update(_supervised_build(part,z[ixtr]))
            for key,sig in signals.items():
                if key not in out: out[key]=np.empty(n,dtype='float32')
                ind=weeks if key.startswith('logtarget') else ts
                out[key][ixval]=sig[ind]
        return pd.DataFrame(out,index=tr.index)
    def fit(self,tr,z=None,supervised=False):
        self.supervised=supervised
        self.signals=_demand_build(tr)
        if supervised:
            if z is None: raise ValueError('Requiere z solo en ajuste train')
            self.signals.update(_supervised_build(tr,np.asarray(z)))
        return self
    def transform(self,df):
        ts=_slots(df.pickup_datetime)
        _,_,_,weeks=_time_fields(df)
        return pd.DataFrame({k:v[weeks if k.startswith('logtarget') else ts]
                             for k,v in self.signals.items()},index=df.index)
