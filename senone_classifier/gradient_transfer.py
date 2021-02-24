import time

import torch

from torch.utils.data import DataLoader

from dp_gradient_select import DPGradientSelector


class GradientTransfer(object):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __init__(self, dataloader, model, optimizer=None, learning_rate=0.01):
        """
        Train a model, replacing the gradient at each epoch with the DP Median
        of previously calculated gradients.
        :param model: Should inherit from nn.Module
        :param train_loader: Instance of DataLoader
        :param test_loader: Instance of DataLoader
        :param learning_rate: Used by optimizer
        """
        self.dataloader = dataloader
        self.train_loader = dataloader['tr_loader']
        self.test_loader = dataloader['cv_loader']
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optimizer if optimizer else torch.optim.SGD(model.parameters(), learning_rate)
        self.gradients = dict((name, []) for name, _ in model.named_parameters())

    def _get_columns_for_median(self, name):
        """
        Used to unwrap gradients into individual components.
        e.g. The ij-th component of each gradient in the list.
        :param name: Key for self.gradients, of the form layer_name.param_name
        :return: Iterator of gradient components
        """
        for x in zip(*map(torch.flatten, self.gradients[name])):
            yield [y.item() for y in x]

    def evaluate(self):
        """Compute average of scores on each batch (unweighted by batch size)"""
        return torch.mean(torch.tensor([self.model.score(batch) for batch in self.test_loader]), dim=0)

    def train(self, epoch_size=5, sample_limit=None):
        """
        Run training with the gradient transfer process
        :param epoch_size: Number of epochs to train each batch
        :param sample_limit: How many samples to train over
        :return:
        """
        start = time.time()
        batch_size = 5
        sample_limit = sample_limit if sample_limit else len(self.train_loader)
        print(f"Train length: {sample_limit}")
        total_loss = 0.0
        global_index = 0
        for epoch in range(0, epoch_size):
            for i, sample in enumerate(self.train_loader):
                x = sample['features'].cuda()
                y = sample['labels'].cuda()
                self.gradients = dict((name, []) for name, _ in self.model.named_parameters())
                for j in range(0, batch_size):
                    loss = self.model(x, y)
                    loss.backward()

                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            grad = param.grad.detach().clone().to(self.device)
                            self.gradients[name].append(grad)

                for name, grad in self.gradients.items():
                    # Use gradient selector for DP Median selection
                    dp_gradient_selector = DPGradientSelector(self.gradients[name], epsilon=1.0)
                    gradient_result = dp_gradient_selector.another_select_gradient_tensor()
                    gradient = gradient_result['point'].cuda()
                    # print(gradient)
                    # medians = dp_gradient_selector.select_gradient_tensor()
                    # Names are of the form "linear1.weight"
                    layer, param = name.split('.')
                    getattr(getattr(self.model, layer), param).grad = gradient
                self.optimizer.step()
                self.optimizer.zero_grad()
                if i > sample_limit - 1:
                    break
                loss = self.model(x, y)
                total_loss += loss.item()
                global_index += 1
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} |' 
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                            epoch + 1, i + 1, total_loss / (i + 1),
                            loss.item(), 1000 * (time.time() - start) / (global_index + 1)), flush=True)

            # accuracy = self.evaluate()
            # print(f"Epoch: {epoch: 5d} | Accuracy: {accuracy.item():.2f}")  # | Loss: {loss.item():.2f}")

    def run_plain(self, epoch_size=5, sample_limit=None):
        """
        For comparison, run without any gradient transfers
        :return:
        """
        start = time.time()
        sample_limit = sample_limit if sample_limit else len(self.train_loader)
        # optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        total_loss = 0.0
        for epoch in range(epoch_size):
            for i, batch in enumerate(self.train_loader):
                # print(f"Batch {i}")
                x = batch['features'].cuda()
                y = batch['labels'].cuda()
                # print("Calculating Loss")
                loss = self.model(x, y)
                # print("Backward")
                loss.backward()
                # print("Step")
                self.optimizer.step()
                # print("Zero Grad")
                self.optimizer.zero_grad()
                # print("Evaluate")
                if i >= sample_limit:
                    break
                total_loss += loss.item()
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} |' 
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                            epoch + 1, i + 1, total_loss / (i + 1),
                            loss.item(), 1000 * (time.time() - start) / (i + 1)), flush=True)
