from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.rtc.modeling_rtc import RTCConfig, RTCAttentionSchedule
from lerobot.eval.robot import RobotInferenceServer
# from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

config = PreTrainedConfig.from_pretrained("/mnt/data/sftp/data/vla/vr_checkpoints/pi0-drawer-placing-2026-05-13_01-57-21/checkpoints/last/pretrained_model")
config.rtc_config = RTCConfig(
    enabled=True,
    execution_horizon=10,
    max_guidance_weight=5.0,
    prefix_attention_schedule=RTCAttentionSchedule.EXP,
    debug=False,
)
model = PI0Policy.from_pretrained("/mnt/data/sftp/data/vla/vr_checkpoints/pi0-drawer-placing-2026-05-13_01-57-21/checkpoints/last/pretrained_model",
                                  config=config)
model = model.cuda()
model.eval()

robot = RobotInferenceServer(model, port=6000)
robot.run()
 