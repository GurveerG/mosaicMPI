
# detect version from package metadata
import importlib.metadata
__version__ = importlib.metadata.version('mosaicmpi')

# get CPU affinity for MP-enabled tasks
import os
if hasattr(os, "sched_getaffinity"):
    cpus_available = len(os.sched_getaffinity(0))
else:
    cpus_available = os.cpu_count()

logging_started = False

from .dataset import Dataset
from .config import Config
from .integration import Integration
from .network import Network
from .colors import Colors
from .plots import *
from .utils import start_logging

import sys
import os
from types import SimpleNamespace
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(".."))) # Add the src directory to sys.path

from mosaicmpi.dataset import Dataset
from mosaicmpi.config import Config
from mosaicmpi.integration import Integration
from mosaicmpi.network import Network
from mosaicmpi.colors import Colors
from mosaicmpi.plots import *
from mosaicmpi.utils import start_logging

dataset = Dataset.from_h5ad("/Users/gurveergill/Downloads/cNMF_5_30_5_ST.h5ad")
program_df = dataset.get_programs()