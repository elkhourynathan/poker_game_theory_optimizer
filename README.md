# Poker Game Theory Optimizer

## Overview

Developed a poker game theory optimizer in Python for poker video analytics. This project involved creating a sophisticated video analytics pipeline using a custom-trained YOLOv8 model and ByteTrack to accurately distinguish and track player cards from community cards. Additionally, a Monte Carlo simulation was implemented to calculate the probabilities of winning, tying, and losing, which enables strategic decisions on whether to fold, call, or raise based on the expected value.

## Installation

If you want to run the Poker Game Theory Optimizer locally, follow these steps:

1. Clone the repository:
``` bash
git clone https://github.com/elkhourynathan/poker_game_theory_optimizer.git
```
2. Install the required packages:
``` bash
pip install -r requirements.txt
```
## Usage

1. Use the default input or add your own input inside the `data` directory.
2. Run the main script:
``` bash
python main.py
```
3. View the annotated output, which is saved to the `output_videos` directory.
