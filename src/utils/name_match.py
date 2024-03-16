from src.learners.baseline.er import ERLearner
from src.learners.baseline.ocm import OCMLearner
from src.learners.baseline.onpro import OnProLearner
from src.learners.baseline.derpp import DERppLearner
from src.learners.baseline.er_ace import ER_ACELearner
from src.learners.baseline.gsa import GSALearner

from src.learners.ccldc.erccl import ERCCLLearner
from src.learners.ccldc.eraceccl import ER_ACECCLLearner
from src.learners.ccldc.ocmccl import OCMCCLLearner
from src.learners.ccldc.derppccl import DERppCCLLearner
from src.learners.ccldc.gsaccl import GSACCLLearner
from src.learners.ccldc.onproccl import OnProCCLLearner

from src.buffers.reservoir import Reservoir
from src.buffers.protobuf import ProtoBuf
from src.buffers.SVDbuf import SVDbuf
from src.buffers.greedy import GreedySampler
from src.buffers.fifo import QueueMemory
from src.buffers.boostedbuf import BoostedBuffer
from src.buffers.mlbuf import MLBuf
from src.buffers.indexed_reservoir import IndexedReservoir
from src.buffers.logits_res import LogitsRes
from src.buffers.mgi_reservoir import MGIReservoir


learners = {
    'ER':   ERLearner,
    'OCM': OCMLearner,
    'DERPP': DERppLearner,
    'ERACE': ER_ACELearner,
    'GSA': GSALearner,
    'OnPro': OnProLearner,
    'ERCCLDC': ERCCLLearner,
    'ERACECCLDC': ER_ACECCLLearner,
    'OCMCCLDC': OCMCCLLearner,
    'DERPPCCLDC': DERppCCLLearner,
    'GSACCLDC': GSACCLLearner,
    'OnProCCLDC': OnProCCLLearner,
    }

buffers = {
    'reservoir': Reservoir,
    'protobuf': ProtoBuf,
    'svd': SVDbuf,
    'greedy': GreedySampler,
    'logits_res': LogitsRes, 
    'fifo': QueueMemory,
    'boost': BoostedBuffer,
    'mlbuf': MLBuf,
    'idx_reservoir': IndexedReservoir,
    'mgi_reservoir': MGIReservoir
}
