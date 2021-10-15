import os
import time
import torch
try:
    from opendp.network.odometer_reconstruction import ReconstructionPrivacyOdometer
    from opendp.network.odometer_stochastic import StochasticPrivacyOdometer
    from opendp.network.odometer_manual import ManualPrivacyOdometer
except ImportError as e:
    print("Install opendp from the external-sgd branch here: https://github.com/opendp/opendp/tree/external-sgd")
    raise e


class Trainer(object):
    def __init__(self, dataloader, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.tr_loader = dataloader['tr_loader']
        self.cv_loader = dataloader['cv_loader']

        self.accountant = None
        if args.step_epsilon:
            # full grad reconstruction odometer
            # self.odometer = ReconstructionPrivacyOdometer(step_epsilon=args.step_epsilon)
            # self.model = self.odometer.make_tracked_view(self.model)

            # # stochastic odometer
            self.odometer = StochasticPrivacyOdometer(step_epsilon=args.step_epsilon)
            self.odometer.track_(self.model)

            # # manual odometer
            # self.odometer = ManualPrivacyOdometer(model=self.model, step_epsilon=args.step_epsilon)

        self.federation = args.federation if hasattr(args, 'federation') else {}

        # Training config
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop

        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path

        # logging
        self.print_freq = args.print_freq

        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        self._reset()

    def _reset(self):
        if self.continue_from:
            print("Loading checkpoint model %s" % self.continue_from)
            package = torch.load(self.continue_from)
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        # create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("training...")
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-'*85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, tr_avg_loss))
            print('-'*85)

            # save model after each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            print("Cross validation...")

            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            if self.odometer:
                self.odometer.increment_epoch()

            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # learning rate halving
            if self.half_lr and val_loss >= self.prev_val_loss:
                if self.early_stop and self.halving:
                    print("even after halving learning rate, "
                          "improvement is too small. Stop training")
                    break
                self.halving = True
            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                        optim_state['param_groups'][0]['lr'] / 2.0
                self.optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
            self.prev_val_loss = val_loss

            # Save best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, sample in enumerate(data_loader):
            x = sample['features'].cuda()
            y = sample['labels'].cuda()
            #print(x.shape)
            loss = self.model(x, y)
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()

                # enable only for ManualPrivacyOdometer
                # self.odometer.privatize_grad()
                self.optimizer.step()
            total_loss += loss.item()
            if (self.federation.get('rank') == 0 or not self.federation) and i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} |'
                        ' Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                            epoch + 1, i + 1, total_loss / (i + 1),
                            loss.item(), 1000 * (time.time() - start) / (i + 1)),
                        flush=True)
        return total_loss / (i + 1)




