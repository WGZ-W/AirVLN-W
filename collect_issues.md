

# Collect Issues


## 1. msgpack_numpy 序列化和反序列化失败
```
Traceback (most recent call last):
  File "/mnt/sdd/weiguanzhao/AirVLN_ws/AirVLN-W/src/vlnce_src/test_lmdb.py", line 55, in <module>
    inspect_lmdb(lmdb_directory, max_keys=3)
  File "/mnt/sdd/weiguanzhao/AirVLN_ws/AirVLN-W/src/vlnce_src/test_lmdb.py", line 31, in inspect_lmdb
    data = msgpack_numpy.unpackb(value, raw=False)
  File "/mnt/sdc/weiguanzhao/miniconda3/envs/AirVLN/lib/python3.10/site-packages/msgpack_numpy.py", line 287, in unpackb
    return _unpackb(packed, **kwargs)
  File "/mnt/sdc/weiguanzhao/miniconda3/envs/AirVLN/lib/python3.10/site-packages/msgpack/fallback.py", line 136, in unpackb
    ret = unpacker._unpack()
  File "/mnt/sdc/weiguanzhao/miniconda3/envs/AirVLN/lib/python3.10/site-packages/msgpack/fallback.py", line 636, in _unpack
    ret.append(self._unpack(EX_CONSTRUCT))
  File "/mnt/sdc/weiguanzhao/miniconda3/envs/AirVLN/lib/python3.10/site-packages/msgpack/fallback.py", line 659, in _unpack
    ret[key] = self._unpack(EX_CONSTRUCT)
  File "/mnt/sdc/weiguanzhao/miniconda3/envs/AirVLN/lib/python3.10/site-packages/msgpack/fallback.py", line 661, in _unpack
    ret = self._object_hook(ret)
  File "/mnt/sdc/weiguanzhao/miniconda3/envs/AirVLN/lib/python3.10/site-packages/msgpack_numpy.py", line 103, in decode
    return np.ndarray(buffer=obj[b'data'],
TypeError: buffer is too small for requested array
```

尝试使用 pickle 替换 msgpack_numpy


## 2. 原始的数据集没有rgb图像和指令
修改程序，新加上原始rgb图像和str指令。

## 3. 模拟器获取不到图像 
取消程序，重新生成，同时把 LMDB 库中的记录删除

发现现象，多个显卡驱动程序的时候，就会获得这个错误？？？