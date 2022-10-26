
from config import *
def re_train():
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    mx = -1
    for action in actions:
        print(action)
        for sequence in range(length_of_action[action]):
            window = []
            dir_frame = os.listdir(os.path.join('D:/BTL_ML_DL/MP_DATA/dataset', action, str(sequence)))
            for frame_num in range(len(dir_frame)):
                print(os.path.join('D:/BTL_ML_DL/MP_DATA/dataset', action, str(sequence), "{}.npy".format(frame_num)))
                try:
                    res = np.load(os.path.join('D:/BTL_ML_DL/MP_DATA/dataset', action, str(sequence), "{}.npy".format(frame_num)))
                    window.append(res)
                except:
                    pass
            
            for i in range (0, len(window), 10):
                w = []
                if i + 10 >= len(window):
                    break
                for j in range(i, i + 10, 1):
                    w.append(window[j])
                sequences.append(w)
                labels.append(label_map[action])

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='sigmoid', input_shape=(10,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    model.fit(X_train, y_train, epochs=10, callbacks=[tb_callback])
    model.save('my_model.h5')

re_train()