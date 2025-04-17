from .lanl_loader import LANLLoader
from .optc_loader import OpTCLoader

LOADERS = {
    "OPTC": OpTCLoader,
    "LANL": LANLLoader,
}
