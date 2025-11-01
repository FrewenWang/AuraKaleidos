#!/bin/bash

# CREDIT: https://github.com/pjreddie/darknet/tree/master/scripts/get_coco_dataset.sh
# shellcheck disable=SC2034

# Clone COCO API
git clone https://github.com/pdollar/coco
cd coco

mkdir images
cd images

# 定义变量
TRAIN2014_URL="https://pjreddie.com/media/files/train2014.zip"
VAL2014_URL="https://pjreddie.com/media/files/val2014.zip"

TRAIN2014_HASH="abcd1234efgh5678ijkl91011mnop1213qrst1415uvw1617xyz1819202122" # 预期的哈希值
VAL2014_HASH="abcd1234efgh5678ijkl91011mnop1213qrst1415uvw1617xyz1819202122" # 预期的哈希值

TRAIN2014_FILE="train2014.zip"
VAL2014__FILE="train2014.zip"

DOWNLOAD_TRAIN2014="false"
DOWNLOAD_VAL2014="false"


function download_train2014() {
  # 检查文件是否存在
  if [[ -f "$TRAIN2014_FILE" ]]; then
      echo "文件 $TRAIN2014_FILE 已存在，校验完整性..."
      # 计算文件的 sha256 哈希值
      ACTUAL_HASH=$(sha256sum "$TRAIN2014_FILE" | awk '{print $1}')

      # 比较哈希值
      if [[ "$ACTUAL_HASH" == "$EXPECTED_HASH" ]]; then
          echo "文件完整，无需重新下载。"
      else
          echo "文件不完整或已损坏，重新下载..."
          rm -f "$FILE"
          DOWNLOAD_TRAIN2014="true"
      fi
  else
      echo "文件 $FILE 不存在，准备下载..."
      DOWNLOAD_TRAIN2014="true"
  fi

  if [[ "$DOWNLOAD_TRAIN2014" == "true" ]]; then
      # Download Images
      # wget -c "$TRAIN2014_URL"
      curl -o "$TRAIN2014_FILE" "$TRAIN2014_URL"
  fi
}


# Unzip
unzip -q train2014.zip
unzip -q val2014.zip

cd ..

# Download COCO Metadata
wget -c https://pjreddie.com/media/files/instances_train-val2014.zip
wget -c https://pjreddie.com/media/files/coco/5k.part
wget -c https://pjreddie.com/media/files/coco/trainvalno5k.part
wget -c https://pjreddie.com/media/files/coco/labels.tgz
tar xzf labels.tgz
unzip -q instances_train-val2014.zip

# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt
