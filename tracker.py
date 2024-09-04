import cv2

# img_file = 'car.jpg'
video = cv2.VideoCapture('London_cam.mp4')

# Pre trained car classifier
classifier_file = 'cars2.xml'
pedestrian_classifier = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_classifier)

# Runs forever
while True:
  read_successful, frame = video.read()

  # Safe coding.
  if read_successful:
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  else:
    break

  cars = car_tracker.detectMultiScale(grayscaled_frame)
  for (x, y, w, h) in cars:
    cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255,0,0), 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)


  pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
  for (x, y, w, h) in pedestrians:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


  cv2.imshow('Car Detector', frame)

  key = cv2.waitKey(1)

  if key==81 or key==113:
    break

video.release()

# 4:09:28