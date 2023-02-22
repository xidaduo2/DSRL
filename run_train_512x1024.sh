# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#Configs
DATA_PATH=D:/xidaduo/DATASETS/CityScapes
OUTPUT_PATH=train_espnetv2_dsrl_512x1024_output
WEIGHT_FILE=ckpt-segmentation/espnetv2_dsrl/512x1024/espnetv2_2.0_2048_best.pth
#
#echo "========Step1: Perform Testing========"
CUDA_VISIBLE_DEVICES=0 python train.py --model espnetv2_dsrl --s 2.0 --dataset cityscapes --data-path ${DATA_PATH} --savedir ${OUTPUT_PATH} --ckpt-file ${WEIGHT_FILE}


#
##if [ ! -d "./gtFine_val_gt" ]; then
##  mkdir ./gtFine_val_gt/
##fi
#
##echo "========Step2: Collect all gt files========"
###cp ${DATA_PATH}/gtFine/val/**/*_labelIds.png ./gtFine_val_gt/
##cp ${DATA_PATH}/gtFine/val/**/*_labelTrainIds.png ./gtFine_val_gt/
###cp -r ${DATA_PATH}/gtFine/val/ ./gtFine_val_gt/
#
#echo "========Step3: Start Evaluation========"
##python utils/evaluate_miou.py --task segmentation --gt ./gtFine_val_gt/ --result ${OUTPUT_PATH} --result_suffix 'leftImg8bit.png' --gt_suffix 'gtFine_trainIds.png' --num_classes 19 --ignore_label 255 --result_file 'espnetv2_dsrl_512x1024_accuracy.txt'
#python utils/evaluate_miou.py --task segmentation --gt ./gtFine_val_gt/ --result ${OUTPUT_PATH} --result_suffix 'leftImg8bit.png' --gt_suffix 'gtFine_labelTrainIds.png' --num_classes 19 --result_file 'espnetv2_dsrl_512x1024_accuracy.txt'


