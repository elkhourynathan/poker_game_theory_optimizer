from ultralytics import YOLO

model = YOLO('model/best.pt')

results = model.predict('data/poker_hand_3hidden.png', conf=0.4)

print(results[0])

print("=====================================")

for box in results[0].boxes:
    print(box)