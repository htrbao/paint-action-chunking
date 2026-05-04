from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.eval.robot import RobotInferenceServer
# from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

model = PI0Policy.from_pretrained("/home/locht1/vr_checkpoints/pi0_base_aloha_3_tasks_0120_100k")
model = model.cuda()
model.eval()

robot = RobotInferenceServer(model, port=6000)
robot.run()
 