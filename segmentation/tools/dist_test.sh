CONFIG=$1
GPUS=$2
#CHECKPOINT='/raid/ligq/lzx/mmsegmentation/tools/work_dirs/fpn_SDT_512x512_384_ade20k/iter_160000.pth'
#     $CHECKPOINT \
#in_file=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/cal_firing_num.py \
    $CONFIG \
    --launcher pytorch \
    ${@:4}
