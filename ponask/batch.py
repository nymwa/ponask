class Batch:

    def __init__(
            self,
            ei,
            eo = None,
            el = None,
            epm = None):

        self.inputs = ei
        self.outputs = eo
        self.lengths = el
        self.padding = epm

    def __len__(self):
        return self.inputs.shape[1]

    def get_num_tokens(self):
        return sum(self.lengths)

    def cuda(self):
        self.inputs = self.inputs.cuda()

        if self.outputs is not None:
            self.outputs = self.outputs.cuda()

        if self.padding is not None:
            self.padding = self.padding.cuda()

        return self

