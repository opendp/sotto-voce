import torch
import torch.nn.functional as F


class FcNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out, layers):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        self.in_size = D_in
        self.hid_size = H
        self.out_size = D_out
        self.hidden_layers = layers

        super(FcNet, self).__init__()
        self.lstm = torch.nn.LSTM(self.in_size, self.hid_size, self.hidden_layers)
        self.linear1 = torch.nn.Linear(self.hid_size, self.out_size)


    def forward(self, x, y):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        lstm_out, _ = self.lstm(x.view(x.shape[1], 1, -1))
        # swap batch axis to front
        y_pred = F.log_softmax(self.linear1(lstm_out.view(lstm_out.shape[1], lstm_out.shape[0], -1)), dim=2)
        y = torch.squeeze(y, 0)
        
        loss = torch.nn.NLLLoss(reduction='sum')
        y = y.long().unsqueeze(0)
        # always have a batch size of one
        output = loss(y_pred[0], y[0])
        return output

    def recognize(self, x):
        lstm_out, _ = self.lstm(x.view(x.shape[1], 1, -1))
        y_pred = F.softmax(self.linear1(lstm_out.view(x.shape[1], -1)), dim=1)
        #_, y_pred_tags = torch.max(y_pred, dim = 1)
        return y_pred

    def get_acc_utt(self, y_pred, y):
        y = y.view(y.shape[1])
        correct_pred = (y_pred == y).float()
        #acc = correct_pred.sum() / len(correct_pred)
        #acc = torch.round(acc) * 100
        return correct_pred.sum(), len(correct_pred)



    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # encoder
            'input': model.in_size,
            'hidden': model.hid_size,
            'output':model.out_size,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


