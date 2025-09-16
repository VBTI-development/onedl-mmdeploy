#!/bin/sh

set -e
# print env
#python3 tools/check_env.py
backend=${1:-ort}
device=${2:-cpu}
current_dir=$(cd `dirname $0`; pwd)
mmdeploy_dir=$current_dir/../..
cd $mmdeploy_dir

work_dir=$mmdeploy_dir/work_dir_mask-rcnn
mkdir -p $work_dir $mmdeploy_dir/data/tiny_coco

model_cfg=$work_dir/mask-rcnn_r50_fpn_1x_coco.py
checkpoint=$work_dir/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
# https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth
sdk_cfg=configs/mmdet/instance-seg/instance-seg_sdk_dynamic.py
input_img=tests/data/tiger.jpeg

python3 -m mim download onedl-mmdetection --config mask-rcnn_r50_fpn_1x_coco --dest $work_dir

if [ $backend == "ort" ]; then
    deploy_cfg=configs/mmdet/instance-seg/instance-seg_onnxruntime_dynamic.py
    model=$work_dir/end2end.onnx
elif [ $backend == "trt" ]; then
    deploy_cfg=configs/mmdet/instance-seg/instance-seg_tensorrt-fp16_dynamic-320x320-1344x1344.py
    model=$work_dir/end2end.engine
else
  echo "Unsupported Backend=$backend"
  exit
fi

echo "------------------------------------------------------------------------------------------------------------"
echo "deploy_cfg=$deploy_cfg"
echo "model_cfg=$model_cfg"
echo "checkpoint=$checkpoint"
echo "device=$device"
echo "------------------------------------------------------------------------------------------------------------"

python3 tools/deploy.py \
  $deploy_cfg \
  $model_cfg \
  $checkpoint \
  $input_img \
  --device $device \
  --work-dir $work_dir \
  --dump-info

if [ $backend == "trt" ]; then
    echo "Running onnx2tensorrt"
    python3 tools/onnx2tensorrt.py \
    $deploy_cfg \
    $work_dir/end2end.onnx \
    $work_dir/temp
fi

# # prepare dataset
wget -nc -P data/tiny_coco/ https://github.com/lizhogn/tiny_coco_dataset/archive/refs/heads/master.zip
unzip -n data/tiny_coco/master.zip -d data/tiny_coco/

# change dataset location
sed -i "s;data/coco;data/tiny_coco/tiny_coco_dataset-master/tiny_coco;g" $model_cfg

echo "------------------------------------------------------------------------------------------------------------"
echo "Running test with converted model"
echo "Running test with $backend"
echo "------------------------------------------------------------------------------------------------------------"

python3 tools/test.py \
  $deploy_cfg \
  $model_cfg \
  --model $model \
  --device $device \
  --log2file $work_dir/test_ort.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 1

echo "------------------------------------------------------------------------------------------------------------"
echo "Running test with SDK"
echo "------------------------------------------------------------------------------------------------------------"

# change topk for test
sed -i 's/"topk": 5/"topk": 1000/g' $work_dir/pipeline.json

python3 tools/test.py \
  $sdk_cfg \
  $model_cfg \
  --model $work_dir \
  --device $device \
  --log2file $work_dir/test_sdk.log \
  --speed-test \
  --log-interval 50 \
  --warmup 20 \
  --batch-size 1

# test profiler
echo "------------------------------------------------------------------------------------------------------------"
echo "Profile sdk model"
echo "------------------------------------------------------------------------------------------------------------"

python3 tools/profiler.py \
  $sdk_cfg \
  $model_cfg \
  ./data/tiny_coco/tiny_coco_dataset-master \
  --model $work_dir \
  --device $device \
  --batch-size 1 \
  --shape 640x640

echo "All done"
