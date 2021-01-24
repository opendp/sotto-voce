import argparse
import torch
from data_io_utt import *
from kaldiio import WriteHelper

#1272-141231-0002 -c 1 -t PCM_01 -f 0-213360 -o 1272-141231-0002 -spkr 1272

def recognize(args):
    model = FcNet(args.input_dim, args.fc_nodes, args.output_dim, args.hidden_layers)  # input, hidden, output
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(model)
    model.to(device)

    #model = FcNet(args.input_dim, args.fc_nodes, args.output_dim)
    #print(model)
    package = torch.load(args.model_path,  map_location=torch.device('cuda'))
    model.load_state_dict(package['state_dict'])
    model.eval()
    #model.cuda()

    eval_file_details = {'scp_dir':args.eval_scp_path,'scp_file':args.eval_scp_file_name,'label_scp_file':args.eval_label_scp_file_name}
    eval_dataset = SenoneClassification(eval_file_details)

    eval_dataloader = data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=100)
    tot_acc = 0
    tot_utt = 0
    with WriteHelper('ark,scp:file.ark,file.scp') as writer:
        with torch.no_grad():
            for i, sample in enumerate(eval_dataloader):
                print('(%d/%d) decoding' %
                      (i, len(eval_dataloader)), flush=True)
                x = sample['features'].cuda()
                y = sample['labels'].cuda()
                name = sample['name'][0]
                #spkr = name.split("-")[0]
                pred, pred_tags = model.recognize(x)
                acc = model.get_acc_utt(pred_tags.cuda(), y)
                tot_acc += acc
                tot_utt += 1
                writer(name, pred.data.cpu().numpy())

    acc_perc = float(tot_acc/tot_utt)    
    print("Average accuracy:" + str(acc_perc))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model file created by training')
    parser.add_argument('--eval_scp_file_name', type=str, required=True)
    parser.add_argument('--eval_label_scp_file_name', type=str, required=True)
    parser.add_argument('--eval_scp_path', type=str, required=True)

    # Model params
    parser.add_argument("--input_dim", default=100, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--fc_nodes", default=1024, type=int)
    parser.add_argument("--fc_layers", default=1, type=int)
    parser.add_argument("--hidden_layers", default=3, type=int)

    args = parser.parse_args()
    recognize(args)


