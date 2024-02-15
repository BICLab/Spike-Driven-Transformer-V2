# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import torch
from mmengine.model import revert_sync_batchnorm
from mmdet.apis.inference import _preprare_data
from mmdet.apis import init_detector as init_model
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from mmengine.config import Config, DictAction
from mmengine.registry import MODELS
import json
import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def main():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default='/raid/ligq/lzx/mmsegmentation/tools/work_dirs/out/tmp', help='Path to output file')
    parser.add_argument('--in_file',
                        default='/raid/ligq/lzx/data/ADE20k/ADEChallengeData2016/images/validation',
                        help='Path to val file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
             'If specified, it will be automatically saved '
             'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # build the model from a config file and a checkpoint file
    model = init_model(cfg, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    print("Successful Build model.")


    firing_dict = {}
    def forward_hook_fn(module, input, output):
        firing_dict[module.name] = output.detach()

    for n, m in model.named_modules():
        if isinstance(m, MultiStepLIFNode):
            m.name = n
            m.register_forward_hook(forward_hook_fn)

    # init the firing_dict
    T = getattr(model, "T", 1)
    fr_dict, nz_dict = {}, {}
    for i in range(T):
        fr_dict["t" + str(i)] = {}
        nz_dict["t" + str(i)] = {}

    # using 100 images to cal the output
    # img = build_dataset(args.in_file, 100, args)
    import os
    img_dir = []
    imgs = os.listdir(args.in_file)

    for img in imgs:
        img = os.path.join(args.in_file, img)
        data, is_batch = _preprare_data(img, model)
        data_preprocessor = cfg.data_preprocessor

        if isinstance(data_preprocessor, dict):
            data_preprocessor = MODELS.build(data_preprocessor)

        data = data_preprocessor(data, False)
        # import pdb;  pdb.set_trace()
        last_idx = len(data) - 1
        test_num, start = 100, 0
        with torch.no_grad():
            if start<test_num:
                start+=1
            else:
                break
            import pdb; pdb.set_trace()
            images = data['inputs'].to(args.device, non_blocking=True)
            output = model(images)

            for t in range(T):
                fr_single_dict = calc_firing_rate(
                    firing_dict, fr_dict["t" + str(t)], last_idx, t
                )
                fr_dict["t" + str(t)] = fr_single_dict
                nz_single_dict = calc_non_zero_rate(
                    firing_dict, nz_dict["t" + str(t)], last_idx, t
                )
                nz_dict["t" + str(t)] = nz_single_dict
            firing_dict = {}

            functional.reset_net(model)

    del firing_dict
    # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    non_zero_str = json.dumps(nz_dict, indent=4)
    firing_rate_str = json.dumps(fr_dict, indent=4)
    print("non-sero rate: ")
    print(non_zero_str)
    print("\n firing rate: ")
    print(firing_rate_str)

    exit(0)
    # test a single image
    # result = inference_model(model, img)
    # show the results
    # show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     title=args.title,
    #     opacity=args.opacity,
    #     draw_gt=False,
    #     show=False if args.out_file is not None else True,
    #     out_file=args.out_file)

def build_dataset(img_path, test_num, args=None):
    # import pdb; pdb.set_trace()
    import os
    img_dir = []
    imgs = os.listdir(img_path)
    for img in imgs:
        img_dir.append(os.path.join(args.in_file, img))
    img_dir = img_dir[:test_num]
    return img_dir

def calc_non_zero_rate(s_dict, nz_dict, idx, t):
    for k, v_ in s_dict.items():
        v = v_[t, ...]
        x_shape = torch.tensor(list(v.shape))
        all_neural = torch.prod(x_shape)
        z = torch.nonzero(v)
        if k in nz_dict.keys():
            nz_dict[k] += (z.shape[0] / all_neural).item() / idx
        else:
            nz_dict[k] = (z.shape[0] / all_neural).item() / idx
    return nz_dict

def calc_firing_rate(s_dict, fr_dict, idx, t):
    for k, v_ in s_dict.items():
        v = v_[t, ...]
        if k in fr_dict.keys():
            fr_dict[k] += v.mean().item() / idx
        else:
            fr_dict[k] = v.mean().item() / idx
    return fr_dict


def build_datasets(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # eval transform
    t = []
    # if args.input_size <= 512:
    #     crop_pct = 512 / 256
    # else:
    #     crop_pct = 1.0
    size = (512, 512)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':
    main()
