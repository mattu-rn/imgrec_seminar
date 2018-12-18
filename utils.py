import matplotlib.pyplot as plt
import cv2

def show_img( fname ):
  img = cv2.imread( fname )
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  plt.grid(False)
  plt.imshow( img )