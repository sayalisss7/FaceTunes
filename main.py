from asyncio import Future
import os
import random
import subprocess
import time
from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array

import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r"C:\Users\91826\Desktop\New folder\haarcascade_frontalface_default.xml")
classifier =load_model(r"C:\Users\annar\Desktop\project\Emotion_Detection_CNN\model.h5")


emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
t = 10
x = None
while t:
    mins,secs =divmod ( t,60) 
    timer = '{:02d}:{:02d}'.format(mins, secs)
    print(timer, end="\r")
    time.sleep(1)
    t-=1             
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            labels.append(label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        int(input(a))
        break

    if t==0:
        mood=labels.copy()

print("Emotion labels detected in the 10th second:", mood)


cap.release()
cv2.destroyAllWindows()


import random
import pygame
import webbrowser

# Initialize Pygame mixer
pygame.mixer.init()

# Define a dictionary of songs for each mood
songs = {
    'Happy': [webbrowser.open_new("https://open.spotify.com/playlist/4nd7oGDNgfM0rv28CQw9WQ?si=a20b2e56045245a1")],
     'Sad': [webbrowser.open_new("https://open.spotify.com/playlist/0z5GPu1ZL2ryEmPbTyH0tB?si=74218136deb44aab")],
     'Neutral': [webbrowser.open_new("https://open.spotify.com/playlist/4PFwZ4h1LMAOwdwXqvSYHd?si=210bbd5251444cf8")],
     'Surprise':[webbrowser.open_new("https://open.spotify.com/playlist/7vatYrf39uVaZ8G2cVtEik?si=d1a554f247f1456e")],
}

# Define a function to play a random song from the list of songs for a given mood
def play_song(mood):
    song_list = songs.get(mood)
    if song_list:
        song_file = random.choice(song_list)
        pygame.mixer.music.load(song_file)
        pygame.mixer.music.play()

# Play a song for each mood
play_song('Happy')
play_song('Sad')
play_song('Neutral')
play_song('Surprise')

# Wait for 5 seconds before stopping the song
pygame.time.wait(5000)

# Stop the song
pygame.mixer.music.stop()

# Quit Pygame mixer
pygame.mixer.quit()