from ultralytics import YOLO

model = YOLO('model/best.pt')

results = model.predict('data/poker_hand_3hidden.png', save=True)

print(results[0])

print("=====================================")

for box in results[0].boxes:
    print(box)