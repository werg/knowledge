import torch
import torch.nn as nn

class KnowledgeBase(nn.Module):
    def __init__(self, query_size, value_size, resolution):
        super(KnowledgeBase, self).__init__()
        self.query_size = query_size
        self.resolution = resolution
        self.value_size = value_size

        # shape: resolution^query_size x value_size
        self.storage = torch.randn([resolution for _ in range(query_size)] + [value_size])

        # shape: (resolution**query_size) x value_size
        rows = resolution**query_size
        self.storage_flat_view = self.storage.view(rows, value_size)

        # matrix that acts like a digit system for converting indices to flat view indices
        # dim:
        self.flat_converter = torch.unsqueeze(torch.tensor([self.resolution ** i
                                                            for i in reversed(range(query_size))]), 1)

        nm = [[]]
        for i in range(query_size):
            # this gets used to find all the corners of the surrounding hypercube
            # (where the corners' coordinates are integers) for any point
            # we choose a value shy of 0.5, to always make it round
            # towards the edges of the cube we are in
            nm = [[0.4999999] + a for a in nm] + [[-0.4999999] + a for a in nm]

        self.neighbor_map = torch.tensor(nm)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.storage = self.storage.to(*args, **kwargs)
        self.neighbor_map = self.neighbor_map.to(*args, **kwargs)
        rows = self.resolution**self.query_size
        self.storage_flat_view = self.storage.view(rows, self.value_size)
        self.flat_converter = self.flat_converter.to(*args, **kwargs)
        return self

    def forward(self, query):
        with torch.no_grad():
            base = torch.floor(query)
            # this is just base-wrapped, i.e. the values that "stick out" beyond the last index
            # are maintained.
            # unlike modulo, this also works correctly for negative numbers
            scaled_and_wrapped = (query - base) * self.resolution

            # replicate so we can put it through the neighbor map (getting ceil / bot in all permutations)
            replicated_query = torch.unsqueeze(scaled_and_wrapped, 1).expand(-1, 2 ** self.query_size).T
            indices = torch.round(self.neighbor_map + replicated_query)

            # note we don't do modulo until here in order to wrap around the right way
            # in the wraparound case, retrieve the zeroeth element, but still compute distance weights
            # based on un-wrapped values (see below)
            flat_indices = torch.squeeze(torch.matmul(indices.long(),
                                                      self.flat_converter)) % self.resolution

            values = torch.index_select(self.storage_flat_view, 0, flat_indices)

        # nearest voxel value weights: manhattan distance to corners of hypercube between them
        weights = torch.sum(1 - (indices - scaled_and_wrapped).abs(), dim=1)

        # multiply weights by values and divide by query_size
        result = torch.squeeze(torch.matmul(values.T, weights)) / self.query_size

        return result

    def grow_resolution(self):
        old_resolution = self.resolution
        self.resolution *= 2


class KnowledgeLayer(nn.Module):
    def __init__(self, query_size=40, value_size=80, resolution=4, kb=None):
        super(KnowledgeLayer, self).__init__()
        self.add_module('kb', kb or KnowledgeBase(query_size, value_size, resolution))

    def forward(self, query):
        return self.kb(query)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.kb = self.kb.to(*args, **kwargs)
        return self

class KnowledgeQueryNet(nn.Module):
    def __init__(self, input_size, hidden_size=50, query_size=40, value_size=80, resolution=4, kb=None):
        super(KnowledgeQueryNet, self).__init__()

        knowledge_layer = KnowledgeLayer(query_size, value_size, resolution, kb)
        self.add_module('model', nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, query_size),
            knowledge_layer))

        self.value_size = knowledge_layer.kb.value_size

    def forward(self, input):
        return self.model(input)

    def init_weights(self):
        self.model.apply(init_weights)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.model = nn.Sequential(*[item.to(*args, **kwargs) for item in self.model])
        return self

# TODO: handle batch slice
initrange = 0.1
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.zeros_(m.weight)
        nn.init.uniform_(m.weight, -initrange, initrange)
