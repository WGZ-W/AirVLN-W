import airsim
import numpy as np

# 使用服务端分配的第一个端口（通常为 30000）
client = airsim.VehicleClient(ip="127.0.0.1", port=30000, timeout_value=10)
client.confirmConnection()

responses = client.simGetImages([
    airsim.ImageRequest("front_0", airsim.ImageType.Scene, False, False)
])
if responses:
    print("Image received, shape:", np.array(responses[0].image_data_uint8).shape)
else:
    print("Failed to get image")