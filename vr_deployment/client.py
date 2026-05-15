from lerobot.eval.robot import RobotInferenceClient
import datetime as dt
import numpy as np
client = RobotInferenceClient(port=6000)
client.reset()
while True:
    start_time = dt.datetime.now()
    obs = {
        "observation.state": np.random.rand(1, 7).astype(np.float32),
        "observation.images.color.high": np.random.rand(1, 3, 480, 640).astype(np.float32),
        "observation.images.color.wrist_left": np.random.rand(1, 3, 480, 640).astype(np.float32),
        "observation.images.color.wrist_right": np.random.rand(1, 3, 480, 640).astype(np.float32),
        "task": ["test\n"],
        "inference_delay": 2,
        "prev_chunk_left_over": np.random.rand(50, 7).astype(np.float32),
        "execution_horizon": 5
    }
    
    action = client.predict_action_chunk(obs)
    print((dt.datetime.now() - start_time), action)