import os
import argparse
from pathlib import Path
import numpy as np
import torch
from RetinexNet_Modified import RetinexNet  # type: ignore # import the model made in other python file

parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', default="0", help='GPU ID (-1 for CPU)')
parser.add_argument('--data_dir', dest='data_dir', default='/home/sethupathyp/project/dark_face/image', help='directory storing the test data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='/home/sethupathyp/project/ckpts/', help='directory for checkpoints')
parser.add_argument('--res_dir', dest='res_dir', default='/home/sethupathyp/project/results/', help='directory for saving the results')

args, _ = parser.parse_known_args()

def predict(model):
    data_path = Path(args.data_dir)
    # use pathlib Path.glob instead of glob.glob
    test_low_data_names = sorted(data_path.glob('*.*'))
    print('Number of evaluation images: %d' % len(test_low_data_names))

    model.predict_full(
        test_low_data_names,
        res_dir=args.res_dir,
        ckpt_dir=args.ckpt_dir,
        patch_size=256,    # or whatever tile size you prefer
        stride=256         # same as patch_size for non-overlap, or smaller for overlap
	)

if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the results
        if not os.path.exists(args.res_dir):
            os.makedirs(args.res_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model
        model = RetinexNet().cuda()
        predict(model)
    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
