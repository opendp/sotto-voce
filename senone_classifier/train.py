import argparse
import resource
import traceback
from multiprocessing import Event
from multiprocessing.context import Process
from queue import Queue

import torch.distributed as dist

from data_io_utt import *
from trainer import *

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def run_senone_worker(args, rank=None, size=None, step_limit=None, federation_scheme='shuffle', queue=None,
                      end_event=None):
    tr_file_details = {'scp_dir': args.train_scp_dir, 'scp_file': args.train_scp_file_name,
                       'label_scp_file': args.train_label_scp_file}
    tr_dataset = SenoneClassification(tr_file_details)
    tr_dataloader = data.DataLoader(tr_dataset, batch_size=1, shuffle=True, num_workers=50)

    cv_file_details = {'scp_dir': args.cv_scp_dir, 'scp_file': args.cv_scp_file_name,
                       'label_scp_file': args.cv_label_scp_file}
    cv_dataset = SenoneClassification(cv_file_details)
    cv_dataloader = data.DataLoader(cv_dataset, batch_size=1, shuffle=False, num_workers=50)

    dataloader = {"tr_loader": tr_dataloader, "cv_loader": cv_dataloader}

    model = FcNet(
        args.input_dim,  # input
        args.fc_nodes,  # hidden
        args.output_dim,  # output
        args.hidden_layers)
    model.apply(weights_init)

    print(model)

    if rank:
        args.federation = {
            'rank': rank,
            'size': size,
            'federation_scheme': federation_scheme,
            'step_limit': step_limit,
            'end_event': end_event
        }
    # model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learn_rate)

    trainer = Trainer(dataloader, model, optimizer, args)
    trainer.train()


def init_process(rank, size, fn, kwargs, backend='gloo'):
    """
    Initialize the distributed environment.
    """
    # use this command to kill processes:
    # lsof -t -i tcp:29500 | xargs kill
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    kwargs['rank'] = rank
    kwargs['size'] = size
    try:
        fn(**kwargs)
    except Exception:
        if not kwargs['end_event'].is_set():
            traceback.print_exc()
        kwargs['end_event'].set()


def train_multiprocess(worker, size, **kwargs):
    processes = []
    queue = Queue()

    end_event = Event()

    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, worker, {
            'queue': queue, 'end_event': end_event,
            **kwargs
        }))
        p.start()
        processes.append(p)

    end_event.wait()
    # wait for history to be queued
    time.sleep(1)

    for p in processes:
        p.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path params
    parser.add_argument("--base_dir", default=os.getcwd())

    parser.add_argument("--train_scp_dir")
    parser.add_argument("--train_scp_file_name")
    parser.add_argument("--train_label_scp_file")
    parser.add_argument("--cv_scp_dir")
    parser.add_argument("--cv_scp_file_name")
    parser.add_argument("--cv_label_scp_file")

    # Model params
    parser.add_argument("--input_dim", default=100, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--fc_nodes", default=1024, type=int)
    parser.add_argument("--hidden_layers", default=1, type=int)
    #parser.add_argument("--window_size", default=5, type=int)

    # Training params
    #parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learn_rate", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.95, type=float)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                        help='Halving learning rate when get small improvement')
    parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                        help='Early stop training when halving lr but still get'
                             'small improvement')
    parser.add_argument('--step_epsilon', default=None, type=float,
                        help='Set step_epsilon to enable differentially private learning')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='Federate learning if > 1')

    # save and load model
    parser.add_argument('--save_folder', default='exp/temp',
                        help='Location to save epoch models')
    parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                        help='Enables checkpoint saving of model')
    parser.add_argument('--continue_from', default='',
                        help='Continue from checkpoint model')
    parser.add_argument('--model_path', default='final.pth.tar',
                        help='Location to save best validation model')
    # logging
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Frequency of printing training information')

    args = parser.parse_args()

    if args.num_workers > 1:
        train_multiprocess(
            worker=run_senone_worker,
            size=args.num_workers,
            args=args,
            federation_scheme='shuffle')
    else:
        run_senone_worker(args)
