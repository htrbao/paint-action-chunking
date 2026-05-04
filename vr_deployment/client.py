from lerobot.eval.robot import RobotInferenceClient
import datetime as dt
import numpy as np
client = RobotInferenceClient(port=6000)
client.reset()
while True:
    start_time = dt.datetime.now()
    obs = {
        "observation.state": np.random.rand(1, 14).astype(np.float32),
        "observation.images.color.high": np.random.rand(1, 3, 480, 640).astype(np.float32),
        "observation.images.color.wrist_left": np.random.rand(1, 3, 480, 640).astype(np.float32),
        "observation.images.color.wrist_right": np.random.rand(1, 3, 480, 640).astype(np.float32),
        "task": ["test\n"]
    }
    
    action = client.get_action(obs)
    print((dt.datetime.now() - start_time), action)