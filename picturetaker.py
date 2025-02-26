import cv2
import os
import keyboard

#set folder for where the pictures will be saved
folderDirectory = "/Users/aminnazari/Downloads"

#Naming conventions for the pictures.
def filename(fnum):
    if fnum < 10:
        fname = folderDirectory + "000" + str(fnum) + ".jpg"
    elif fnum < 100:
        fname = folderDirectory + "00" + str(fnum) + ".jpg"
    elif fnum < 1000:
        fname = folderDirectory + "0" + str(fnum) + ".jpg"
    else:
        fname = folderDirectory + str(fnum) + ".jpg"
    return fname

def main():
    print("Welcome, press 'p' to take a picture and 'Ctrl+C' to quit")

    cap = cv2.VideoCapture(0)
    filenum = 0

    # Iterate to the next picture name
    while True:
        try:
            f = open(filename(filenum))
            f.close()
            filenum += 1
        except IOError as e:
            break

    # Loop to capture and display feed
    while cap.isOpened():
        _, frame = cap.read()
        cv2.imshow('frame', frame)

        # Use to capture picture
        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite(filename(filenum), frame)
            filenum += 1  # Corrected indentation

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()  # Properly release the video capture object

if __name__ == '__main__':
    main()