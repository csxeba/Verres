import pafy
import cv2

video_url = "https://www.youtube.com/watch?v=BcC9ZMeRmBk"
print("Parsing with pafy")
player = pafy.new(video_url).getbest()
print("Received", player.url)
cap = cv2.VideoCapture(player.url)
while 1:
    success, frame = cap.read()
    if not success:
        print("Failed :(")
        break
    print("Frame:", frame.shape)
    cv2.imshow("Frames", frame)
    cv2.waitKey(1000//25)
