import numpy as np
import pandas as pd

def haversine_np(lat1, lon1, lat2, lon2):
    """Distancia ortodrómica en km, con NumPy vectorizado."""
    lat1,lon1,lat2,lon2=map(np.radians,(lat1,lon1,lat2,lon2))
    dl=lat2-lat1; dn=lon2-lon1
    a=np.sin(dl/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dn/2)**2
    return 6371.0088*2*np.arcsin(np.sqrt(np.clip(a,0,1)))

def create_features(frame):
    d=frame.copy()
    out=pd.DataFrame(index=d.index)
    dt=pd.to_datetime(d['pickup_datetime'],errors='coerce')
    hour=dt.dt.hour+dt.dt.minute/60+dt.dt.second/3600
    wd=dt.dt.dayofweek
    out['vendor_id']=pd.to_numeric(d.vendor_id,errors='coerce')
    out['passenger_count']=pd.to_numeric(d.passenger_count,errors='coerce')
    out['passenger_invalid']=(out.passenger_count<1).astype('int8')
    out['store_and_fwd_flag']=(d.store_and_fwd_flag=='Y').astype('int8')
    out['month']=dt.dt.month
    out['day']=dt.dt.day
    out['day_of_year']=dt.dt.dayofyear
    out['day_of_week']=wd
    out['week_of_year']=dt.dt.isocalendar().week.astype('float64')
    out['hour']=dt.dt.hour
    out['hour_decimal']=hour
    out['minute']=dt.dt.minute
    out['is_weekend']=wd.isin([5,6]).astype('int8')
    out['is_rush_hour']=((wd<5)&(((hour>=7)&(hour<10))|((hour>=16)&(hour<20)))).astype('int8')
    out['is_night']=((hour<6)|(hour>=22)).astype('int8')
    out['hour_sin']=np.sin(2*np.pi*hour/24)
    out['hour_cos']=np.cos(2*np.pi*hour/24)
    out['weekday_sin']=np.sin(2*np.pi*wd/7)
    out['weekday_cos']=np.cos(2*np.pi*wd/7)
    out['dayyear_sin']=np.sin(2*np.pi*dt.dt.dayofyear/366)
    out['dayyear_cos']=np.cos(2*np.pi*dt.dt.dayofyear/366)
    out['month_sin']=np.sin(2*np.pi*(dt.dt.month-1)/12)
    out['month_cos']=np.cos(2*np.pi*(dt.dt.month-1)/12)
    for c in ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']:
        out[c]=pd.to_numeric(d[c],errors='coerce')
    for prefix in ['pickup','dropoff']:
        lon=out[f'{prefix}_longitude']; lat=out[f'{prefix}_latitude']
        bad=(~lon.between(-74.5,-73.3))|(~lat.between(40.4,41.2))
        out[f'{prefix}_bad_gps']=bad.astype('int8')
        out.loc[bad,[f'{prefix}_longitude',f'{prefix}_latitude']]=np.nan
    lon1=out.pickup_longitude.to_numpy(); lon2=out.dropoff_longitude.to_numpy()
    lat1=out.pickup_latitude.to_numpy(); lat2=out.dropoff_latitude.to_numpy()
    latm=(lat1+lat2)/2
    dy=(lat2-lat1)*111.195
    dx=(lon2-lon1)*111.195*np.cos(np.radians(latm))
    out['dx_km']=dx; out['dy_km']=dy
    out['abs_dx_km']=np.abs(dx);out['abs_dy_km']=np.abs(dy)
    out['haversine_km']=haversine_np(lat1,lon1,lat2,lon2)
    out['manhattan_km']=np.abs(dx)+np.abs(dy)
    out['haversine_log']=np.log1p(out['haversine_km'])
    out['manhattan_log']=np.log1p(out['manhattan_km'])
    out['trip_distance_squared']=out.haversine_km**2
    out['manhattan_ratio']=out.manhattan_km/(out.haversine_km+0.1)
    out['is_near_zero_distance']=(out.haversine_km<0.10).astype('int8')
    out['bearing']=np.arctan2(dx,dy)
    out['bearing_sin']=np.sin(out.bearing)
    out['bearing_cos']=np.cos(out.bearing)
    out['mid_lat']=(lat1+lat2)/2
    out['mid_lon']=(lon1+lon2)/2
    out['lat_lon_start_prod']=lat1*lon1
    out['lat_lon_end_prod']=lat2*lon2
    for prefix in ['pickup','dropoff']:
        lon=out[f'{prefix}_longitude'];lat=out[f'{prefix}_latitude']
        out[f'{prefix}_grid_lat_100']=(lat*100).round()
        out[f'{prefix}_grid_lon_100']=(lon*100).round()
        out[f'{prefix}_grid_lat_200']=(lat*200).round()
        out[f'{prefix}_grid_lon_200']=(lon*200).round()
        out[f'{prefix}_grid_lat_500']=(lat*500).round()
        out[f'{prefix}_grid_lon_500']=(lon*500).round()
        # Las ubicaciones son geografía pública, no se usan destinos posteriores desconocidos.
        for name,alat,alon in [('jfk',40.6413,-73.7781),('lga',40.7769,-73.8740),
                               ('ewr',40.6895,-74.1745),('times_sq',40.7580,-73.9855),
                               ('grand_central',40.7527,-73.9772),('downtown',40.7061,-74.0086)]:
            out[f'{prefix}_dist_{name}_km']=haversine_np(lat.to_numpy(),lon.to_numpy(),alat,alon)
        out[f'{prefix}_near_jfk']=(out[f'{prefix}_dist_jfk_km']<2.5).astype('int8')
        out[f'{prefix}_near_lga']=(out[f'{prefix}_dist_lga_km']<2.5).astype('int8')
        out[f'{prefix}_near_ewr']=(out[f'{prefix}_dist_ewr_km']<3.5).astype('int8')
    out['any_airport']=(out[['pickup_near_jfk','pickup_near_lga','pickup_near_ewr','dropoff_near_jfk','dropoff_near_lga','dropoff_near_ewr']].max(axis=1))
    out['hour_x_distance']=out.hour_decimal*out.haversine_km
    out['rush_x_distance']=out.is_rush_hour*out.haversine_km
    out['weekend_x_distance']=out.is_weekend*out.haversine_km
    # Reducir memoria y normalizar la representación numérica (NO estandarizar estadísticos).
    return out.astype(np.float32).replace([np.inf,-np.inf],np.nan)
