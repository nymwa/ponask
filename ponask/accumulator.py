class Accumulator:

    def __init__(
            self,
            epoch,
            num_batches):

        self.epoch = epoch
        self.num_batches = num_batches

        self.losss = []
        self.wpbs = []
        self.spbs = []
        self.lrs = []
        self.grads = []

    def update(self, batch, loss, lr, grad):
        self.losss.append(loss.item())
        self.wpbs.append(batch.get_num_tokens())
        self.spbs.append(len(batch))
        self.lrs.append(lr)
        self.grads.append(grad)

    def step_log(self):
        line = '| inner'
        line += ' | epoch {}, {}/{}'.format(
                self.epoch,
                len(self.spbs),
                self.num_batches)
        line += ' | loss {:.4f}'.format(self.losss[-1])
        line += ' | lr {:.8f}'.format(self.lrs[-1])
        if self.grads[0] is not None:
            line += ' | grad {:.4f}'.format(self.grads[-1])
        line += ' | w/b {}'.format(self.wpbs[-1])
        line += ' | s/b {}'.format(self.spbs[-1])
        return line

    def avg(self, lst):
        num_examples = sum(self.spbs)
        return sum([n * x for n, x in zip(self.spbs, lst)]) / num_examples

    def epoch_log(self, num_steps):
        line = '| train'
        line += ' | epoch {}'.format(self.epoch)
        line += ' | loss {:.4f}'.format(self.avg(self.losss))
        line += ' | lr {:.8f}'.format(self.avg(self.lrs))
        if self.grads[0] is not None:
            line += ' | grad {:.4f}'.format(self.avg(self.grads))
        line += ' | w/b {:.1f}'.format(self.avg(self.wpbs))
        line += ' | s/b {:.1f}'.format(self.avg(self.spbs))
        line += ' | steps {}'.format(num_steps)
        return line

