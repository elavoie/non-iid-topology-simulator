import torch
import setup.meta as m

def load(rundir):
    topology = m.load(rundir, 'topology.json')
    edges = topology['edges']
    topology['edges'] = { int(rank):edges[rank] for rank in edges }
    topology['weights'] = torch.tensor(topology['weights'])
    return topology

