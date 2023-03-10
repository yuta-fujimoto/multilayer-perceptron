import matplotlib.pyplot as  plt
import argparse
import json
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2)
    for log in args.logs:
        with open(log, 'r') as f:
            history = json.load(f)
            filename = os.path.splitext(os.path.basename(log))

            x = np.arange(1, history['epoch'] + 1)
            axes[0][0].plot(x, history['train_loss'], label=f'train_{filename[0]}', alpha=0.7, linewidth=0.5)
            axes[0][0].plot(x, history['valid_loss'], label=f'validation_{filename[0]}', alpha=0.7, linewidth=0.5)
            axes[0][0].set_xlabel('epoch')
            axes[0][0].set_ylabel('loss')
            axes[0][0].grid(True)
            axes[0][0].set_ylim(0.0, 0.5)

            axes[0][1].plot(x, history['train_accuracy'], alpha=0.7, linewidth=0.5)
            axes[0][1].plot(x, history['valid_accuracy'], alpha=0.7, linewidth=0.5)
            axes[0][1].set_xlabel('epoch')
            axes[0][1].set_ylabel('accuracy')
            axes[0][1].grid(True)
            axes[0][1].set_ylim(0.9, 1.0)

            axes[1][0].plot(x, history['train_false_negative'], alpha=0.7, linewidth=0.5)
            axes[1][0].plot(x, history['valid_false_negative'], alpha=0.7, linewidth=0.5)
            axes[1][0].set_xlabel('epoch')
            axes[1][0].set_ylabel('false negative rate')
            axes[1][0].grid(True)
            axes[1][0].set_ylim(0.0, 0.2)

            axes[1][1].axis('off')

    fig.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig('visualize.png')
