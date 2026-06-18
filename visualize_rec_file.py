import mxnet as mx
import os
import numbers

root_dir = "/workspace/FaceNist/Data"
path_imgrec = os.path.join(root_dir, 'train.rec')
path_imgidx = os.path.join(root_dir, 'train.idx')

imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)

print(f"Flag: {header.flag}")
print(f"Label: {header.label}")
print(f"ID: {header.id}")
print(f"header[0]: {header.label[0]}")
print(f"header[1]: {header.label[1]}")

import numpy as np
keys = imgrec.keys
print("5 first images")
for i in range(1, 7):
    if i >= len(keys):
        break
    idx = keys[i]
    print(f"idx: {idx}")
    s = imgrec.read_idx(idx)
    header, img_data = mx.recordio.unpack(s)
    label = header.label
    if not isinstance(label, numbers.Number):
        label = label[0]
    sample = mx.image.imdecode(img_data).asnumpy()
    print(f"Key/Index: {idx}, Label: {label}, Shape: {sample.shape}")