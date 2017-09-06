from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from Tkinter import *
from tkFileDialog import askopenfilename
from PIL import Image
from PIL import ImageTk
#img = cv2.imread('2.jpg')
root = Tk()
#setting up a tkinter canvas with scrollbars
frame = Frame(root, bd=2, relief=SUNKEN)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)
xscroll = Scrollbar(frame, orient=HORIZONTAL)
xscroll.grid(row=1, column=0, sticky=E+W)
yscroll = Scrollbar(frame)
yscroll.grid(row=0, column=1, sticky=N+S)
canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
canvas.grid(row=0, column=0, sticky=N+S+E+W)
xscroll.config(command=canvas.xview)
yscroll.config(command=canvas.yview)
frame.pack(fill=BOTH,expand=1)
#adding the image
File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
img1 = ImageTk.PhotoImage(Image.open(File))

img = cv2.imread(File)



height, width, dim = img.shape
print(height,width)

#img = img[(height/4):(3*height/4), (width/4):(3*width/4), :]
height, width, dim = img.shape
#cv2.imshow("pic",img)
img_vec = np.reshape(img, [height * width, dim] )

kmeans = KMeans(n_clusters=3)
kmeans.fit( img_vec )

unique_l, counts_l = np.unique(kmeans.labels_, return_counts=True)
sort_ix = np.argsort(counts_l)
sort_ix = sort_ix[::-1]



fig = plt.figure()
#plt.show()
ax = fig.add_subplot(111)
x_from = 0.05
#x=np.array[255,255,255]
#y=np.array[255,255,255]
i=0
for cluster_center in kmeans.cluster_centers_[sort_ix]:
	if(i==1):
		cv2.rectangle(img,(width-100,10),(width-30,40),(cluster_center[0],cluster_center[1], cluster_center[2]),-1)
		font = cv2.FONT_HERSHEY_TRIPLEX
		text = '#%02x%02x%02x' % (cluster_center[2],cluster_center[1], cluster_center[0])
		cv2.putText(img,'Color-Code:',(width-185,55), font, .4,(255,0,0),1)
		cv2.putText(img,'shirt-color:',(width-185,30), font, .4,(255,0,0),1)
		cv2.putText(img,text,(width-100,55), font, .4,(cluster_center[0],cluster_center[1], cluster_center[2]),1)
		print (cluster_center[2],cluster_center[1], cluster_center[0])
		print (text)
		cv2.imshow("editied",img)
		#ax.add_patch(patches.Rectangle((x_from, 0.05), 0.29, 0.9, alpha=None,facecolor='#%02x%02x%02x' % (cluster_center[2],cluster_center[1], cluster_center[0] ) ) )
		#x_from = x_from + 0.31
		break
	    #cv2.Rectangle(img,)
	i=i+1
cv2.waitKey(0)


