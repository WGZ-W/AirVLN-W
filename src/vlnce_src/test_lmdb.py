import lmdb
import msgpack_numpy
import numpy as np
import msgpack
import pickle

import sys
from pathlib import Path

# 假设原项目根目录为 /path/to/AirVLN-W
project_root = Path("/mnt/sdd/weiguanzhao/AirVLN_ws/AirVLN-W")
sys.path.insert(0, str(project_root))  # 使 Model 模块可导入

from Model.utils.tensor_dict import TensorDict, DictTree

import pdb

def inspect_lmdb(lmdb_path, max_keys=5):
    """
    查看 LMDB 中存储的轨迹数据。
    
    Args:
        lmdb_path: LMDB 目录路径
        max_keys: 最多显示多少个键的数据
    """

    # pdb.set_trace()
    # 打开 LMDB（只读，不锁定）
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
    
    with env.begin() as txn:
        print(f"LMDB 中共有 {txn.stat()['entries']} 条记录\n")
        
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= max_keys:
                break
            
            print(f"=== 键: {key.decode()} ===")
            # print(value[:100])
            
            # 反序列化 msgpack 数据
            # data = msgpack_numpy.unpackb(value, raw=False)
            data = pickle.loads(value)
            
            # data 的结构: [obs_dict, prev_actions, oracle_actions]
            obs_dict, prev_actions, oracle_actions = data
            
            print(f"观测字典包含的传感器: {list(obs_dict.keys())}")
            for sensor, arr in obs_dict.items():
                if type(arr) is not str: 
                    print(f"  - {sensor}: shape {arr.shape}, dtype {arr.dtype}")
                else:
                    print(f"  - {sensor}: {arr} ")
            
            print(f"prev_actions 形状: {prev_actions.shape}, 内容前10: {prev_actions[:10]}")
            print(f"oracle_actions 形状: {oracle_actions.shape}, 内容前10: {oracle_actions[:10]}")
            
            # 如果存在指令 tokens，可尝试解码（需要自行实现 decode）
            if 'instruction' in obs_dict:
                instr_tokens = obs_dict['instruction']
                print(f"instruction tokens shape: {instr_tokens.shape}, 示例: {instr_tokens[:10]}")
            
            print()  # 空行分隔
    
    env.close()

if __name__ == "__main__":
    # 替换为你的 LMDB 路径
    # lmdb_directory = "/mnt/sdd/weiguanzhao/AirVLN_ws/DATA/img_features/collect/AirVLN-Test/train_rgb"
    lmdb_directory = "/mnt/sdd/weiguanzhao/AirVLN_ws/DATA/img_features/collect/AirVLN-Test/train"
    inspect_lmdb(lmdb_directory, max_keys=3)

    # import cv2
    # env = lmdb.open(lmdb_directory, readonly=True)
    # with env.begin() as txn:
    #     for key, value in txn.cursor():

    #         print(value[:20])
    #         img_array = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    #         print(img_array.shape)  # (H, W, 3)
    #         cv2.imwrite("output.jpg", img_array)
    #         # 尝试两种常见的反序列化方式
    #         try:
    #             data = pickle.loads(value)
    #         except:
    #             data = value  # 可能是 JPEG 字节串
    #         print(f"Key: {key.decode()}, Type: {type(data)}")
    #         if isinstance(data, np.ndarray):
    #             print(f"  Array shape: {data.shape}, dtype: {data.dtype}")
    #         elif isinstance(data, bytes):
    #             print(f"  Bytes length: {len(data)} (likely encoded image)")
    #         # break

