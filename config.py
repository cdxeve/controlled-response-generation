import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default='AmazonQA')
parser.add_argument("--epoch", type=int, default=80)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--mode", type=str, default='forward')

parser.add_argument("--cuda_device", type=str, default="0")
parser.add_argument("--evaluate_start",type=int,default=0)
parser.add_argument("--evaulate_end",type=int,default=1000)

arg = parser.parse_args()
print(arg)

dataset=arg.dataset
epoch = arg.epoch
batch_size= arg.batch_size
mode = arg.mode
cuda_device = arg.cuda_device


