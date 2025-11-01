#!/bin/bash
# convert caffe models to formats supported by inference framework
# the connverted models will be uploaded to model-hub on the server

# configure the MODEL_HOME path
MODEL_HOME=~/models
NCNN_CVT_TOOL=$MODEL_HOME/tools/ncnn/caffe2ncnn
NCNN_OPT_TOOL=$MODEL_HOME/tools/ncnn/ncnnoptimize
PDLITE_CVT_TOOL=$MODEL_HOME/tools/paddle_lite/auto_transform.sh
PDLITE_QUANT_TOOL=$MODEL_HOME/tools/paddle_lite/quantize/quant_post_paddle.py
PDLITE_OPT_TOOL=$MODEL_HOME/tools/paddle_lite/opt

show_help() {
    echo "Usage: $0 [model_dir] [model_tag]" >&2
    echo
    echo "   [model_dir] is the source model (e.g. caffe models) directory"
    echo "   [model_tag] is the category, e.g. face_rect, or, face_landmark"
    echo
    echo "   -h, --help              show help message"
    echo
}

# parse arguments
if [ "$#" -ge 2 ]; then
    MODEL_PATH="$1"
    MODEL_TAG="$2"
elif [ "$#" -eq 1 ]; then
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_help
        exit 0
    else
        MODEL_PATH="$1"
    fi
fi

if [ -z "$MODEL_PATH" ]; then
    show_help
    exit 1
fi

# check whether the convert tool exists
echo "==> MODEL_HOME=$MODEL_HOME"
echo "==> NCNN_CVT_TOOL=$NCNN_CVT_TOOL"
echo "==> PADDLE_LITE_CVT_TOOL=$PDLITE_CVT_TOOL"
if [ -f "$NCNN_CVT_TOOL" ] || [ -f "$PDLITE_CVT_TOOL" ]; then
    echo "==> model convert tool FOUND!"
else
    echo "==> model convert tool NOT FOUND! STOPPED!"
    exit 1
fi

# convert models
echo "==> model path=$MODEL_PATH"
if [ ! -d "$MODEL_PATH" ]; then
    echo "==> model path DOES NOT EXISTS! STOPPED!"
    exit 1
fi

for file in "$MODEL_PATH"/*
do
    filename=$(basename -- "$file")
    ext="${filename##*.}"
    filename="${filename%.*}"
    if [ "$ext" = "caffemodel" ]; then
        MODEL_WEIGHT="$filename".caffemodel
        MODEL_VERSION="$filename"
    elif [ "$ext" = "prototxt" ]; then
        MODEL_DEF="$filename".prototxt
    fi
done

if [ -z "$MODEL_WEIGHT" ] || [ -z "$MODEL_DEF" ]; then
    echo "==> valid source model files NOT FOUND! STOPPED!"
    exit 1
fi

echo "==> source model verison = $MODEL_VERSION"
echo "==> source model weight file = $MODEL_WEIGHT"
echo "==> source model description file = $MODEL_DEF"

cd "$MODEL_PATH"
mkdir -p "$MODEL_VERSION"/caffe
cp $MODEL_DEF $MODEL_WEIGHT "$MODEL_VERSION"/caffe/

if [ ! -z "$MODEL_TAG" ]; then
    echo "tag: $MODEL_TAG" > "$MODEL_VERSION"/tag
fi

# caffe -> ncnn
# ncnn fp32
mkdir -p "$MODEL_VERSION"/ncnn
$NCNN_CVT_TOOL $MODEL_DEF ${MODEL_WEIGHT} ${MODEL_VERSION}/ncnn/${MODEL_VERSION}.param ${MODEL_VERSION}/ncnn/${MODEL_VERSION}.bin
#$NCNN_OPT_TOOL ${MODEL_VERSION}/ncnn/${MODEL_VERSION}.param ${MODEL_VERSION}/ncnn/${MODEL_VERSION}.bin ${MODEL_VERSION}/ncnn/${MODEL_VERSION}_opt.param ${MODEL_VERSION}/ncnn/${MODEL_VERSION}_opt.bin 0
# ncnn int8
# todo: ncnn-int8

# caffe -> paddle-lite
mkdir -p "$MODEL_VERSION"/paddle_lite
if [ "$(uname)" == "Darwin" ]; then
  cd $MODEL_HOME/tools/paddle_lite
  sh $PDLITE_CVT_TOOL --framework=caffe --prototxt=$MODEL_PATH/$MODEL_DEF \
                      --weight=$MODEL_PATH/$MODEL_WEIGHT \
                      --fluid_save_dir=$MODEL_PATH/$MODEL_VERSION/paddle_lite/fluid \
                      --optimize_out=$MODEL_PATH/$MODEL_VERSION/paddle_lite/$MODEL_VERSION
  cd -
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  cd $MODEL_HOME/tools/paddle_lite
  bash $PDLITE_CVT_TOOL --framework=caffe --prototxt=$MODEL_PATH/$MODEL_DEF \
                      --weight=$MODEL_PATH/$MODEL_WEIGHT \
                      --fluid_save_dir=$MODEL_PATH/$MODEL_VERSION/paddle_lite/fluid \
                      --optimize_out=$MODEL_PATH/$MODEL_VERSION/paddle_lite/$MODEL_VERSION
  cd -
fi
# paddle int8 quantize
python ${PDLITE_QUANT_TOOL} \
  --img_dir $MODEL_HOME/calibration_set/$MODEL_TAG \
  --model_path $MODEL_PATH/$MODEL_VERSION/paddle_lite/fluid/inference_model \
  --save_path $MODEL_PATH/$MODEL_VERSION/paddle_lite/${MODEL_VERSION}_int8

if [ "$(uname)" == "Darwin" ]; then
  PDLITE_OPT_TOOL=${PDLITE_OPT_TOOL}_mac
fi

${PDLITE_OPT_TOOL} \
  --model_file=$MODEL_PATH/$MODEL_VERSION/paddle_lite/${MODEL_VERSION}_int8/__model__ \
  --param_file=$MODEL_PATH/$MODEL_VERSION/paddle_lite/${MODEL_VERSION}_int8/__params__ \
  --valid_targets=arm \
  --optimize_out_type=naive_buffer \
  --optimize_out=$MODEL_PATH/$MODEL_VERSION/paddle_lite/${MODEL_VERSION}_int8

# upload the models to model-hub
# in order to upload models, the host machine should be able to connect to server via ssh using rsa keys
SERVER_IP=172.20.72.11
SERVER_USER=iov
MODEL_HUB_ROOT=/home/iov/ModelHub
MODEL_DEST_DIR="$SERVER_USER"@"$SERVER_IP":"$MODEL_HUB_ROOT"
echo "==> model_hub address = $MODEL_DEST_DIR"

#scp -pr "$MODEL_VERSION" "$MODEL_DEST_DIR"/
echo "==> upload models to model_hub DONE!"

cd -