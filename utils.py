import matplotlib.pyplot as plt
import cv2

def show_img( fname ):
  img = cv2.imread( fname )
  plt.grid(False)
  plt.imshow( img )