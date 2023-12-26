import face_recognition
import cv2
import numpy as np
import mediapipe as mp
import socket
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
def login():
    global name
    cap = cv2.VideoCapture(0)

    Abdalla = face_recognition.load_image_file("./Faces/Facee.jpg")
    AbdallaFR = face_recognition.face_encodings(Abdalla)[0]

    Zoz = face_recognition.load_image_file("./Faces/Zoz.jpg")
    ZozFR = face_recognition.face_encodings(Zoz)[0]

    Rina = face_recognition.load_image_file("./Faces/rina.jpg")
    Rina_FR = face_recognition.face_encodings(Rina)[0]

    Omar = face_recognition.load_image_file("./Faces/Omar.jpg")
    Omar_FR = face_recognition.face_encodings(Omar)[0]

    Samy = face_recognition.load_image_file("./Faces/Samy.jpg")
    Samy_FR = face_recognition.face_encodings(Samy)[0]


    known_face_encodings = [
        AbdallaFR,
        Rina_FR,
        ZozFR,
        Omar_FR,
        Samy_FR
    ]
    known_face_names = [
        "Abdalla",
        "Rina",
        "Zoz",
        "Omar",
        "Samy"
    ]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    while cap.isOpened():
        ret, frame = cap.read()

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)


        process_this_frame = not process_this_frame


        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

mySocket = socket.socket()
mySocket.bind(('localhost', ))
mySocket.listen(5)
conn , addr = mySocket.accept()
print("device connected")
login()
while True:
    msg =bytes(name, 'utf-8')
    conn.send(msg)
    break