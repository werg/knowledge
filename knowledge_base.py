import torch

class KnowledgeBase(object):
    def __init__(self, query_size, value_size, resolution):
        self.query_size = query_size
        self.resolution = resolution

        # shape: resolution^query_size x value_size
        self.storage = torch.randn([resolution for _ in range(query_size)] + [value_size])

        # shape: (resolution**query_size) x value_size
        self.storage_flat_view = self.storage.view(resolution**query_size, value_size)

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

    def query(self, query):
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
        

# grow size iteratively / increase_resolutionn (interpolation)
# store new values
