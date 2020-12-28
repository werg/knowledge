import torch
import torch.nn as nn

class KnowledgeBase(nn.Module):
    def __init__(self, query_size, value_size, resolution, device=None):
        super(KnowledgeBase, self).__init__()
        self.query_size = query_size
        self.resolution = resolution
        self.value_size = value_size

        # shape: resolution^query_size x value_size
        self.register_buffer('storage', torch.randn([resolution for _ in range(query_size)] + [value_size], device=device, dtype=torch.half))

        # shape: (resolution**query_size) x value_size
        rows = resolution**query_size
        self.register_buffer('storage_flat_view', self.storage.view(rows, value_size))

        # matrix that acts like a digit system for converting indices to flat view indices
        # dim:
        self.register_buffer('flat_converter', torch.unsqueeze(torch.tensor([float(self.resolution ** i)
                                                                             for i in reversed(range(query_size))], device=device), 1))
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.storage = self.storage.to(*args, **kwargs)
        rows = self.resolution**self.query_size
        self.storage_flat_view = self.storage.view(rows, self.value_size)
        self.flat_converter = self.flat_converter.to(*args, **kwargs)
        return self

    def forward(self, query):
        base = torch.floor(query)
        # this is just base-wrapped, i.e. the values that "stick out" beyond the last index
        # are maintained.
        # unlike modulo, this also works correctly for negative numbers
        scaled_and_wrapped = (query - base) * self.resolution


        with torch.no_grad():

            anchor_index = torch.round(scaled_and_wrapped)
            # get the direction of deviation from anchor_index in every dimension
            deviation = scaled_and_wrapped - anchor_index
            simplex_directions = torch.sign(deviation)
            # the diagonal contains the direction in which each vertex of simplex deviates from anchor
            diag_directions = torch.diag_embed(simplex_directions)
            # add zero for the anchor
            padded_directions = F.pad(diag_directions, (0,0, 0,1), "constant", 0)
            anchor_replicated = anchor_index.view(-1, 1, self.query_size).expand_as(padded_directions)

            indices = anchor_replicated + padded_directions

            # note we don't do modulo until here in order to wrap around the right way
            # in the wraparound case, retrieve the zeroeth element, but still compute distance weights
            # based on un-wrapped values (see below)
            flat_indices = ((torch.matmul(indices,
                                         self.flat_converter).long()) % self.resolution).view(-1)

            values = torch.index_select(self.storage_flat_view, 0, flat_indices).view(-1, self.query_size + 1, self.value_size)

        # nearest voxel value weights: manhattan distance to corners of hypercube between them
        weights = torch.sum(1 - (indices - scaled_and_wrapped.view(-1,1,self.query_size).expand_as(indices)).abs(), dim=2)

        # multiply weights by values and divide by query_size
        result = torch.squeeze(torch.bmm(weights.unsqueeze(1), values.float()) , dim=1)/ self.query_size

        return result

    def grow_resolution(self):
        old_resolution = self.resolution
        self.resolution *= 2


class KnowledgeLayer(nn.Module):
    def __init__(self, query_size=40, value_size=80, resolution=4, kb=None):
        super(KnowledgeLayer, self).__init__()
        self.value_size = value_size
        self.kb = kb or KnowledgeBase(query_size, value_size, resolution)

    def forward(self, query):
        return self.kb(query)
        #return query

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.kb = self.kb.to(*args, **kwargs)
        return self


class MultiHeadKnowledgeLayer(nn.Module):
    def __init__(self, heads, query_size=40, value_size=80, resolution=4, kb=None):
        super(MultiHeadKnowledgeLayer, self).__init__()
        self.heads = heads
        self.value_size = value_size
        self.kb = kb or KnowledgeBase(query_size, value_size, resolution)
        self.query_size = query_size

    def forward(self, query):
        query = query.view(-1, self.query_size)
        return self.kb(query).view(-1, self.value_size * self.heads)


class KnowledgeQueryNet(nn.Module):
    def __init__(self, input_size, hidden_size=50, query_size=40, value_size=80, resolution=4, kb=None):
        super(KnowledgeQueryNet, self).__init__()

        self.knowledge_layer = KnowledgeLayer(query_size, value_size, resolution, kb)
        self.add_module('preprocess', nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, query_size)))

        #self.value_size = self.knowledge_layer.kb.value_size
        self.value_size = value_size

    def forward(self, input):
        return self.knowledge_layer(self.preprocess(input))

    def init_weights(self):
        self.preprocess.apply(init_weights)

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.preprocess = nn.Sequential(*[item.to(*args, **kwargs) for item in self.preprocess])
        self.knowledge_layer.to(*args, **kwargs)
        return self

# TODO: handle batch slice
initrange = 0.1
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.zeros_(m.weight)
        nn.init.uniform_(m.weight, -initrange, initrange)
