from card_tracker import CardTracker
from utils import read_image, read_video, save_video
from poker import PokerEngine
from treys import Card


class PokerPipeline:
    def __init__(self):
        self.poker_engine = PokerEngine()
    
    def run(self):
        card_tracker = CardTracker('model/best.pt')
        image_frame = read_image('data/poker_hand_3hidden.png')

        tracks = card_tracker.get_object_tracks(image_frame)

        organized_card_data = card_tracker.detect_cards_from_tracks(tracks)

        cards_with_simulations = card_tracker.assign_probabilities_to_frames(organized_card_data, self.poker_engine)

        output_video_frames = card_tracker.annotate_frames(image_frame, cards_with_simulations)

        save_video(output_video_frames, 'output_videos/output.mp4')
