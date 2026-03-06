import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    """-------------------------Training--------------------------"""
    parser.add_argument('--seed', type=int, default=42,
                        help="number of seed")
    parser.add_argument('--bs', type=int, default=4,
                        help="number of batch size")
    parser.add_argument('--epochs', type=int, default=50,
                        help="number of train epoch")
    parser.add_argument('--lr', type=float, default=3e-4,
                        help="learning rate")
    parser.add_argument('--wd', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help="Max Grad Norm")
    parser.add_argument('--use_cls', type=bool, default=False,
                        help="Use cls")
    parser.add_argument('--thr', type=float, default=0.5,
                        help="BCE Threshold")
    parser.add_argument('--fold', type=int, default=5,
                        help="Num of Fold")
    parser.add_argument('--num_classes', type=int, default=1,
                        help="Num of Classes")
    parser.add_argument('--D', type=int, default=128,
                        help="Dimension")    
    args = parser.parse_args()
    return args