import cv2
import insightface
import numpy as np
import time

app = insightface.app.FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

database = np.load('../data/faces_database.npy', allow_pickle=True).item()

cap = cv2.VideoCapture(0)

print("--- JETSON NANO ATTENDANCE SYSTEM ACTIVE ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    faces = app.get(frame)
    
    for face in faces:
        emb = face.embedding
        name = "Unknown"
        max_sim = 0
        
        for person, db_emb in database.items():
            sim = np.dot(emb, db_emb) / (np.linalg.norm(emb) * np.linalg.norm(db_emb))
            if sim > 0.45 and sim > max_sim: 
                name = person
                max_sim = sim
        
        bbox = face.bbox.astype(int)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, f"{name}", (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if name != "Unknown":
            print(f"ATTENDANCE MARKED: {name} at {time.strftime('%H:%M:%S')}")

    cv2.imshow('Jetson Attendance System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()