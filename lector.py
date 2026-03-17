import rioxarray as rxr

ds = rxr.open_rasterio('MCD43A3.A2026055.h10v08.061.2026068142620.hdf')
print(ds)