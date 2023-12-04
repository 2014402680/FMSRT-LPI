import torch
import argparse
from mask import Mask
from utils import get_data, set_seed
from model import GNNEncoder, EdgeDecoder, DegreeDecoder, GMAE
# main parameter
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=2, help="Choose Datasets (1 or 2)")
parser.add_argument('--seed', type=int, default=2023, help="Random seed for model and dataset.")
parser.add_argument('--dim', type=int, default=1050, help='Feature Dimension of Similarity Matrix(dataset1 >= 990, dataset2 >= 1050)')
parser.add_argument('--alpha', type=float, default=0.007, help='loss weight for degree prediction.')
parser.add_argument('--p', type=float, default=0.4, help='Mask ratio')
args = parser.parse_args()
set_seed(args.seed)

# dataset1
# {'AUC': '0.998797', 'AP': '0.992417', 'ACC': '0.998195', 'SEN': '1.000000', 'PRE': '0.996403', 'SPE': '0.996390', 'F1': '0.998198', 'MCC': '0.996396'}
# {'AUC': '0.990508', 'AP': '0.983418', 'ACC': '0.981949', 'SEN': '0.993983', 'PRE': '0.970623', 'SPE': '0.969916', 'F1': '0.982164', 'MCC': '0.964178'}
# dataset2
# {'AUC': '0.998883', 'AP': '0.993429', 'ACC': '0.999441', 'SEN': '1.000000', 'PRE': '0.998883', 'SPE': '0.998881', 'F1': '0.999441', 'MCC': '0.998882'}
# {'AUC': '0.991613', 'AP': '0.987032', 'ACC': '0.975391', 'SEN': '0.977629', 'PRE': '0.973274', 'SPE': '0.973154', 'F1': '0.975446', 'MCC': '0.950793'}
splits = get_data(args.dataset, args.dim)

encoder = GNNEncoder(in_channels=args.dim, hidden_channels=64, out_channels=128)
edge_decoder = EdgeDecoder(in_channels=128, hidden_channels=64, out_channels=2)
degree_decoder = DegreeDecoder(in_channels=128, hidden_channels=64, out_channels=1)
mask = Mask(p=args.p)

model = GMAE(encoder, edge_decoder, degree_decoder, mask).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
for epoch in range(1000):
    model.train()
    loss = model.train_epoch(splits['train'], optimizer, alpha=args.alpha)
model.eval()
test_data = splits['test']
z = model.encoder(test_data.x, test_data.edge_index)

test_auc, test_ap, acc, sen, pre, spe, F1, mcc = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index)
results = {'AUC': "{:.6f}".format(test_auc),
            'AP': "{:.6f}".format(test_ap),
            "ACC": "{:.6f}".format(acc),
            "SEN": "{:.6f}".format(sen),
            "PRE": "{:.6f}".format(pre),
            "SPE": "{:.6f}".format(spe),
            "F1": "{:.6f}".format(F1),
            "MCC": "{:.6f}".format(mcc)}
print(results)
