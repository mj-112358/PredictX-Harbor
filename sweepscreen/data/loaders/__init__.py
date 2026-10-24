"""Dataset loaders. Each ``load(raw_dir, extracted_dir)`` returns a list of :class:`~sweepscreen.data.schema.Record`."""
from . import breast, breastvid, busbra, busi, busuclm, udiat

#: site name -> (loader module, folder under data/raw)
LOADERS = {
    busi.SITE: (busi, "busi"),
    busbra.SITE: (busbra, "busbra"),
    breast.SITE: (breast, "breast"),
    udiat.SITE: (udiat, "udiat"),
    busuclm.SITE: (busuclm, "busuclm"),
    breastvid.SITE: (breastvid, "breastvid"),
}

#: Sites made of still images (one record per image); BreastVid is the video site.
STILL_SITES = (busi.SITE, busbra.SITE, breast.SITE, udiat.SITE, busuclm.SITE)
VIDEO_SITES = (breastvid.SITE,)
