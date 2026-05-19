import argparse
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.rtc.modeling_rtc import RTCConfig, RTCAttentionSchedule
from lerobot.common.policies.repaint.modeling_repaint import RepaintConfig
from lerobot.eval.robot import RobotInferenceServer

CHECKPOINT_PATH = "/mnt/data/sftp/data/vla/vr_checkpoints/pi0-drawer-placing-2026-05-13_01-57-21/checkpoints/last/pretrained_model"

def parse_args():
    parser = argparse.ArgumentParser(description="PI0 Robot Inference Server")
    parser.add_argument(
        "--smooth-option",
        type=str,
        default="",
        help="",
    )
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--execution-horizon", type=int, default=10)
    return parser.parse_args()


def build_config(args) -> PreTrainedConfig:
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if args.smooth_option == "rtc":
        config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.max_guidance_weight,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
            debug=False,
        )
    elif args.smooth_option == "repaint":
        config.repaint_config = RepaintConfig(
            enabled=True,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
            debug=False,
        )
    return config


def main():
    args = parse_args()
    print(f"Smooth (RTC): {args.smooth}")

    config = build_config(args)
    model = PI0Policy.from_pretrained(args.checkpoint, config=config)
    model = model.cuda().eval()

    robot = RobotInferenceServer(model, port=args.port)
    robot.run()


if __name__ == "__main__":
    main()