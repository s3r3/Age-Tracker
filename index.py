import cv2
from deepface import DeepFace


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    
    try:
        analysis = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
        
        if isinstance(analysis, list):
            for face in analysis:
                age = face["age"]
                region = face["region"]
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'Age: {int(age)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        else:
            age = analysis["age"]
            region = analysis["region"]
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Age: {int(age)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    except Exception as e:
        print(f"Error: {e}")

    cv2.imshow('Age Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
