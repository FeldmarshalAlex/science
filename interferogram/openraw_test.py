from rawkit.raw import Raw
from rawkit.options import WhiteBalance

with Raw(filename='original\\160119-3_nulevaya.cr2') as raw:
    raw.options.white_balance = WhiteBalance(camera=False, auto=True)
    raw.save(filename='image.ppm')