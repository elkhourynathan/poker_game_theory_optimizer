import cv2
from ultralytics import YOLO
import supervision as sv
from poker import PokerEngine



class CardTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        self.frame_width = 0
        self.frame_height = 0
    
    def detect_frames(self, frames):
        batch_size = 5
        detections = []
        for i in range(0, len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detection_batch
        return detections

    def get_object_tracks(self, frames):

        self.frame_height, self.frame_width = frames[0].shape[:2]

        detections = self.detect_frames(frames)

        tracks = {'cards' : []}

        for frame_num, detection in enumerate(detections):

            cls_names = detection.names

            # Convert to supervision format

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track card objects

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['cards'].append({})
            if detection_supervision:
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]

                    card_class = cls_names[cls_id]
                    tracks['cards'][frame_num][track_id] = {"bbox": bbox, "class": card_class}
        
        return tracks

    def detect_cards_from_tracks(self, tracks):
        organized_card_data = []

        for frame_num, frame_tracks in enumerate(tracks['cards']):
            card_position = {}

            # Group by card class
            for track_id, detection in frame_tracks.items():
                card_class = detection['class']
                bbox = detection['bbox']

                if card_class not in card_position:
                    card_position[card_class] = []
                card_position[card_class].append({'bbox': bbox, 'track_id': track_id})
            
            # Create bounding box around entire card
            full_bboxes = []
            for card_class, detections in card_position.items():
                if len(detections) == 2:
                    top_left_bbox = min(detections, key=lambda x: x['bbox'][1])
                    bottom_right_bbox = max(detections, key=lambda x: x['bbox'][3])
                    full_bbox = [
                        top_left_bbox['bbox'][0], top_left_bbox['bbox'][1], # x1, y1
                        bottom_right_bbox['bbox'][2], bottom_right_bbox['bbox'][3] # x2, y2
                    ]
                    track_id = top_left_bbox['track_id']  # Assuming the same track_id for both corners
                else:
                    full_bbox = detections[0]['bbox']
                    track_id = detections[0]['track_id']
                
                full_bboxes.append({'bbox': full_bbox, 'class': card_class, 'frame_id': frame_num, 'track_id': track_id})
            
            if not full_bboxes:
                continue

            middle_y_with_buffer = self.frame_height // 2

            frame_data = {
                'frame_id': frame_num,
                'user_cards': [],
                'community_cards': [],
                'middle_y_with_buffer': middle_y_with_buffer
            }

            for card in full_bboxes:
                card_info = {'bbox': card['bbox'], 'class': card['class'], 'track_id': card['track_id']}
                if card['bbox'][1] > middle_y_with_buffer:
                    frame_data['user_cards'].append(card_info)
                else:
                    frame_data['community_cards'].append(card_info)
            
            organized_card_data.append(frame_data)

        return organized_card_data


    def annotate_frames(self, frames, card_information):
        output_video_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            # Get cards for the current frame
            frame_data = card_information[frame_num]

            for card in frame_data['user_cards']:
                bbox = card['bbox']
                card_class = card['class']
                color = (0, 255, 0)  # Green for user cards
                frame = self.draw_rectangle(frame, bbox, color, card_class)

            for card in frame_data['community_cards']:
                bbox = card['bbox']
                card_class = card['class']
                color = (255, 0, 0)  # Red for community cards
                frame = self.draw_rectangle(frame, bbox, color, card_class)
            

            # Draw the middle_y_with_buffer line for the current frame
            middle_y_with_buffer = frame_data['middle_y_with_buffer']
            frame = self.draw_buffer_line(frame, middle_y_with_buffer)

            # Draw user and community cards text annotation
            user_cards = [card['class'] for card in frame_data['user_cards']]
            community_cards = [card['class'] for card in frame_data['community_cards']]
            frame = self.draw_card_information(frame, user_cards, community_cards)

            # Draw probability information
            print(frame_data)
            if 'probability_data' in frame_data:
                probability_data = frame_data['probability_data']
                probability_text = f"Win: {probability_data['win_prob']:.2f}, Tie: {probability_data['tie_prob']:.2f}, Loss: {probability_data['loss_prob']:.2f}"
                frame = self.draw_probability_information(frame, probability_text)

            output_video_frames.append(frame)
        
        return output_video_frames
    
    def assign_probabilities_to_frames(self, card_information, poker_engine: PokerEngine):

        for card_frame in card_information:
            user_cards = [x['class'] for x in card_frame['user_cards']]
            community_cards = [x['class'] for x in card_frame['community_cards']]
            if len(user_cards) != 2:
                continue

            # Set the player hand and community cards for the poker engine
            poker_engine.set_player_hand(user_cards)
            poker_engine.set_community_cards(community_cards)

            # Run monte carlos sim
            win_prob, tie_prob, loss_prob = poker_engine.monte_carlo_simulation()

            probability_data = {
                'win_prob': win_prob,
                'tie_prob': tie_prob,
                'loss_prob': loss_prob
            }
            card_frame['probability_data'] = probability_data
        
        return card_information
            


    def draw_rectangle(self, frame, bbox, color, card_class):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 6)
        cv2.putText(frame, card_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
        return frame
    
    def draw_buffer_line(self, frame, middle_y_with_buffer):
        print(middle_y_with_buffer)
        frame_height, frame_width = frame.shape[:2]
        cv2.line(frame, (0, middle_y_with_buffer), (frame_width, middle_y_with_buffer), (0, 0, 255), 3)
        cv2.putText(frame, 'Buffer Line', (10, middle_y_with_buffer - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
    
    def draw_card_information(self, frame, user_cards, community_cards):
        frame_height, frame_width = frame.shape[:2]
        text_y = frame_height - 60  # Starting y position for the text
        text_x = 10  # Starting x position for the text
        
        user_cards_text = "User Cards: " + ", ".join(user_cards)
        community_cards_text = "Community Cards: " + ", ".join(community_cards)
        
        cv2.putText(frame, user_cards_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(frame, community_cards_text, (text_x, text_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        return frame
    
    def draw_probability_information(self, frame, probability_data):
        frame_height, frame_width = frame.shape[:2]
        text_y = frame_height - (frame_height - 60)  # Starting y position for the text
        text_x = 10  # Starting x position for the text
        
        
        cv2.putText(frame, probability_data, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        return frame

