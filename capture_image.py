# The following makes the app.
# importing open cv library for image/video processing
import cv2
from tkinter import filedialog
import tkinter as thinker

print("Welcome! This app will capture images from the video being recorded by your computer's webcam or take given number of images.")

mode_of_capture = input(
    "Would you like to capture frames from a video or images? Press 'V' for video and 'I' for images: ")
print(mode_of_capture)

# This will return video from the first webcam on the computer.
camera = cv2.VideoCapture(0)

if(mode_of_capture.lower() == 'v'):
    result, image = camera.read()

    # Ask the user for where the image files should be saved.
    print("Select the folder where the images would be saved.")
    file_path = filedialog.askdirectory(title="Select a path")
    count = 0

    # while there is a video result
    while result is True:
        # Selecting the file path
        root = thinker.Tk()
        root.withdraw()

        # create a jpg image with count number as the file name
        cv2.imwrite(file_path + "/vidframe%d.jpg" % count, image)
        # ret is a boolean regarding whether or not there was a return at all,
        # at the frame is each frame that is returned. If there is no frame, cap.read()
        # return None

        # update both variables
        result, image = camera.read()

        # Results printed and count updated, and the image is shown
        print('Reading a new frame: ', result)
        count += 1
        cv2.imshow('image', image)

        # Runs once per frame and if the user presses key q, break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
elif(mode_of_capture.lower() == 'i'):
    # This will return video from the first webcam on the computer.
    camera = cv2.VideoCapture(0)

    # initializing variables
    result, image = camera.read()
    count = 0

    # Ask the user for where the image files should be saved.
    print("Select the folder where the images would be saved.")
    file_path = filedialog.askdirectory(title="Select a path for classes")

    n = input("Enter the number of images. It cannot be greater than 20: ")
    for i in range(min(int(n), 20)):
        return_value, image = camera.read()
        cv2.imwrite(file_path + '/image' + str(i) + '.jpg', image)
        print('Taking a new image: ', return_value)
        cv2.imshow('image', image)

else:
    print("Invalid selection! Try again!")

camera.release()
cv2.destroyAllWindows()
