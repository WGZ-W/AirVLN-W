import json
from collections import defaultdict

# 读取原始JSON文件
with open(f'/mnt/sdd/weiguanzhao/AirVLN_ws/DATA/data/aerialvln/train.json', 'r') as f:
    data = json.load(f)

# 按 scene_id 分组
groups = defaultdict(list)
for episode in data['episodes']:
    scene_id = episode['scene_id']
    groups[scene_id].append(episode)

# 分别写入不同的文件，保持原始结构 {"episodes": [...]}
for scene_id, episodes in groups.items():
    filename = f'/mnt/sdd/weiguanzhao/AirVLN_ws/DATA/data/aerialvln/scene_{scene_id}.json'
    with open(filename, 'w') as f:
        json.dump({"episodes": episodes}, f, indent=2)

print(f"拆分完成，共生成 {len(groups)} 个场景文件。")