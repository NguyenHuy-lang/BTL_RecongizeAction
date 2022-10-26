from config import *
model = keras.models.load_model('D:/BTL_ML_DL/action2.h7')

sequence = []
sentence = []
threshold = 0.7

zezo = [0 for x in  range(0, 21*3 * 2)] 
print(zezo)


# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    path_video = "D:/name_you.mp4"
    cap = cv2.VideoCapture(path_video)
    ret, frame = cap.read()
    while ret:
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        if (keypoints[-21*3 * 2:] == np.array(zezo)).all():
            print("empty nhaaaaaa")
            cv2.imshow('OpenCV Feed', image)
            ret,frame = cap.read()
            continue
        sequence.insert(0,keypoints)
        sequence = sequence[:10]
        
        if len(sequence) == 10:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
        # print(res[np.argmax(res)])
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    print("ok")
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
        if len(sentence) > 5: 
            sentence = sentence[-5:]
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)
        ret,frame = cap.read()
        # Break gracefully
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
