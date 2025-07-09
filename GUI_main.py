import tkinter as tk
from PIL import Image, ImageTk
import time
from tkvideo import tkvideo


##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Suspicious Activity Detection In Exam")

video_label =tk.Label(root)
video_label.pack()
# read video to display on label
player = tkvideo("studentvideo.mp4", video_label,loop = 1, size = (w, h))
player.play()
# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image

label_l1 = tk.Label(root, text="Suspicious Activity Detection In Exam",font=("Times New Roman", 30, 'bold'),
                    background="#00688B", fg="white", width=70, height=2)
label_l1.place(x=0, y=0)

def log():
    from subprocess import call
    call(["python","Log.py"])
    root.destroy()

def reg():
    from subprocess import call
    call(["python","registration.py"])
    root.destroy()
    
    
def window():
    root.destroy()
    
# For Buttons on frame
button1 = tk.Button(root, text="Login", command=log, width=15, height=1,font=('times', 15, ' bold '), bg="#3CB371", fg="white")
button1.place(x=300, y=350)
def on_enter(e):
  button1['background'] = '#FF7D40'

def on_leave(e):
  button1['background'] = '#3CB371'

button1.bind("<Enter>", on_enter)
button1.bind("<Leave>", on_leave)

button2 = tk.Button(root, text="Registration",command=reg,width=15, height=1,font=('times', 15, ' bold '), bg="#3CB371", fg="white")
button2.place(x=500,y=350)
def on_enter(e):
  button2['background'] = '#FF7D40'

def on_leave(e):
  button2['background'] = '#3CB371'

button2.bind("<Enter>", on_enter)
button2.bind("<Leave>", on_leave)

button3 = tk.Button(root, text="Exit",command=window,width=14, height=1,font=('times',15, ' bold '), bg="#8470FF", fg="white")
button3.place(x=400, y=450)
def on_enter(e):
  button3['background'] = '#EEC900'

def on_leave(e):
  button3['background'] = '#8470FF'

button3.bind("<Enter>", on_enter)
button3.bind("<Leave>", on_leave)

root.mainloop()