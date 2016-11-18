<font size=4><b>Reproduced Fashion Style 128 on Fashion 144k dataset</b></font>

contact: anenbergb (anenberg@cs.stanford.edu)

<b>How to run (copied from resnet):</b>

```shell
# cd to the your workspace.
# It contains an empty WORKSPACE file, resnet codes and cifar10 dataset.
ls -R
  .:
  cifar10  resnet  WORKSPACE

  ./cifar10:
  test.bin  train.bin  validation.bin

  ./resnet:
  BUILD  cifar_input.py  g3doc  README.md  resnet_main.py  resnet_model.py

# Build everything for GPU.
bazel build -c opt --config=cuda resnet/...

# Train the model.
bazel-bin/resnet/resnet_main --train_data_path=cifar10/train.bin \
                             --log_root=/tmp/resnet_model \
                             --train_dir=/tmp/resnet_model/train \
                             --dataset='cifar10' \
                             --num_gpus=1

# Evaluate the model.
# Avoid running on the same GPU as the training job at the same time,
# otherwise, you might run out of memory.
bazel-bin/resnet/resnet_main --eval_data_path=cifar10/test.bin \
                             --log_root=/tmp/resnet_model \
                             --eval_dir=/tmp/resnet_model/test \
                             --mode=eval \
                             --dataset='cifar10' \
                             --num_gpus=0
```
