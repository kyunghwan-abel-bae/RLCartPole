import argparse
from dqn_trainer import DQNTrainer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--max_steps', type=int, default=2000)
parser.add_argument('--render_fps', type=int, default=10)
parser.add_argument('--load_dir', type=str, default='checkpoints')

args = parser.parse_args()

trainer = DQNTrainer(
    max_steps=args.max_steps,
    save_dir=args.load_dir
)

trainer.load(args.checkpoint)

trainer.play(
    render_fps=args.render_fps
)