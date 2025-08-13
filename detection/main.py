import time
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

def process_video(opt):
   t0_all = time.time()
   videoSrcPath = opt.source
   if not os.path.exists(videoSrcPath): 
      print(f" Exit as the video path {videoSrcPath} doesnt exist")
      return
   cap = cv2.VideoCapture(videoSrcPath)
   frames_count, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), cap.get(
    cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
   width = int(width)
   height = int(height)
   print(f"Input video #frames ={frames_count}, fps ={fps}, width ={width}, height={height}")

   frameNumber = 0
   cv2.namedWindow("output", cv2.WINDOW_AUTOSIZE)

   while cap.isOpened():
      ret , frame = cap.read()
      if ret:
         #resize if required
         image = np.copy(frame)
         cv2.putText(image, "Frame#: " + str(frameNumber), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (2,10,200), 2)

         cv2.imshow('output', image)
         key = cv2.waitKey(1)
         # Quit when 'q' is pressed
         if key == ord('q'):
            break
         elif key == ord('k'):
            cv2.waitKey(0)
      else:
         print(f"coudn't read current frame #{frameNumber}")
      frameNumber = frameNumber + 1
   cap.release()
   cv2.destroyAllWindows()
   t1_all = time.time()
   print(f'Done. process_video took ({t1_all - t0_all:.3f}s)')


def get_arguments():
    parser = argparse.ArgumentParser(description='program to open a video and display; ',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     usage ='\n #1: open a single video: >> python3 main.py -s "videoname.MP4"')
    
    parser.add_argument('--source', "-s", type=str, required=True, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')# not used 640 for now

    return parser.parse_args()

if __name__ == "__main__":
  args = get_arguments()
  print(args)
  process_video(args)
  print("## Exit out of the program .......")