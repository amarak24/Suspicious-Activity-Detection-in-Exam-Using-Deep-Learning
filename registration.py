import tkinter as tk
from tkinter import messagebox as ms
import sqlite3
from PIL import Image, ImageTk
import re
import numpy

##############################################+=============================================================
root = tk.Tk()
root.configure(background="white")
root.geometry("1300x700") #


w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Suspicious Activity Detection In Exam")

# ++++++++++++++++++++++++++++++++++++++++++++
#####For background Image
image2 = Image.open('7.jpg')
image2 = image2.resize((950,800), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=600, y=0) 


label_l1 = tk.Label(root, text="Suspicious Activity Detection In Exam",font=("Times New Roman", 30, 'bold'),
                    background="#607B8B", fg="white", width=70, height=2)
label_l1.place(x=0, y=0)

######################### For Registration form #####################################################################

Fullname = tk.StringVar()
address = tk.StringVar()
username = tk.StringVar()
Email = tk.StringVar()
Phoneno = tk.IntVar()
password = tk.StringVar()
password1 = tk.StringVar()

# database code
db = sqlite3.connect('evaluationn.db')
cursor = db.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS registration"
               "(Fullname TEXT, address TEXT, username TEXT, Email TEXT, Phoneno TEXT, password TEXT)")
db.commit()



def password_check(passwd): 
	
	SpecialSym =['$', '@', '#', '%'] 
	val = True
	
	if len(passwd) < 6: 
		print('length should be at least 6') 
		val = False
		
	if len(passwd) > 20: 
		print('length should be not be greater than 8') 
		val = False
		
	if not any(char.isdigit() for char in passwd): 
		print('Password should have at least one numeral') 
		val = False
		
	if not any(char.isupper() for char in passwd): 
		print('Password should have at least one uppercase letter') 
		val = False
		
	if not any(char.islower() for char in passwd): 
		print('Password should have at least one lowercase letter') 
		val = False
		
	if not any(char in SpecialSym for char in passwd): 
		print('Password should have at least one of the symbols $@#') 
		val = False
	if val: 
		return val 

def insert():
    fname = Fullname.get()
    addr = address.get()
    un = username.get()
    email = Email.get()
    mobile = Phoneno.get()
    pwd = password.get()
    cnpwd = password1.get()

    with sqlite3.connect('evaluationn.db') as db:
        c = db.cursor()

    # Find Existing username if any take proper action
    find_user = ('SELECT * FROM registration WHERE username = ?')
    c.execute(find_user, [(username.get())])
    

    # to check mail
    regex='^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    if (re.search(regex, email)):
        a = True
    else:
        a = False
    # validation
    if (fname.isdigit() or (fname == "")):
        ms.showinfo("Message", "please enter valid name")
    elif (addr == ""):
        ms.showinfo("Message", "Please Enter Address")
    elif (email == "") or (a == False):
        ms.showinfo("Message", "Please Enter valid email")
    elif((len(str(mobile)))<10 or len(str((mobile)))>10):
        ms.showinfo("Message", "Please Enter 10 digit mobile number")
    elif (c.fetchall()):
        ms.showerror('Error!', 'Username Taken Try a Diffrent One.')
    elif (pwd == ""):
        ms.showinfo("Message", "Please Enter valid password")
    elif(pwd=="")or(password_check(pwd))!=True:
        ms.showinfo("Message", "password must contain atleast 1 Uppercase letter,1 symbol,1 number")
    elif (pwd != cnpwd):
        ms.showinfo("Message", "Password Confirm password must be same")
    else:
        conn = sqlite3.connect('evaluationn.db')
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO registration(Fullname, address, username, Email, Phoneno, password) VALUES(?,?,?,?,?,?)',
                (fname, addr, un, email, mobile, pwd))

            conn.commit()
            db.close()
            ms.showinfo('Success!', 'Account Created Successfully !')
            window.destroy()
            
            from subprocess import call
            call(["python", "Log.py"])
            window.destroy()

#####################################################################################################################################################

# ------------------------- For Registration Frame 


frame_alpr = tk.LabelFrame(root, text=" --Register-- ", width=600, height=702, bd=5, font=('times', 14, ' bold '),fg="white",bg="#607B8B")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=0, y=90)


l1 = tk.Label(frame_alpr, text="Registration Form", font=("Times new roman", 30, "bold","italic"),bd=5, bg="#4D4D4D", fg="white")
l1.place(x=120, y=10)

# that is for label1 registration

l2 = tk.Label(frame_alpr, text="Full Name :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l2.place(x=100, y=100)
t1 = tk.Entry(frame_alpr, textvar=Fullname, width=20, font=('', 15))
t1.place(x=300, y=100)
# that is for label 2 (full name)


l3 = tk.Label(frame_alpr, text="Address :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l3.place(x=100, y=150)
t2 = tk.Entry(frame_alpr, textvar=address, width=20, font=('', 15))
t2.place(x=300, y=150)
# that is for label 3(address)


l5 = tk.Label(frame_alpr, text="E-mail :", width=12, font=("Times new roman", 15, "bold"), bd=5,fg="black")
l5.place(x=100, y=200)
t4 = tk.Entry(frame_alpr, textvar=Email, width=20, font=('', 15))
t4.place(x=300, y=200)
# that is for email address


l6 = tk.Label(frame_alpr, text="Phone number :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l6.place(x=100, y=250)
t5 = tk.Entry(frame_alpr, textvar=Phoneno, width=20, font=('', 15))
t5.place(x=300, y=250)
# phone number

l4 = tk.Label(frame_alpr, text="User Name :", width=12, font=("Times new roman", 15, "bold"), bd=5,fg="black")
l4.place(x=100, y=300)
t3 = tk.Entry(frame_alpr, textvar=username, width=20, font=('', 15))
t3.place(x=300, y=300)
# that is for label 4()

l9 = tk.Label(frame_alpr, text="Password :", width=12, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l9.place(x=100, y=350)
t9 = tk.Entry(frame_alpr, textvar=password, width=20, font=('', 15), show="*")
t9.place(x=300, y=350)


l10 = tk.Label(frame_alpr, text="Confirm Password:", width=13, font=("Times new roman", 15, "bold"),bd=5, fg="black")
l10.place(x=100, y=400)
t10 = tk.Entry(frame_alpr, textvar=password1, width=20, font=('', 15), show="*")
t10.place(x=300, y=400)


btn = tk.Button(frame_alpr, text="Register", bg="#FAEBD7",font=("times new roman",20,"bold"),fg="black", width=9, height=1, bd=5,command=insert)
btn.place(x=220, y=470)

 
 # ------------------ Function for button
 
def log():
    from subprocess import call
    call(["python","Log.py"])
    root.destroy()
    
def con():
    from subprocess import call
    call(["python","GUI_main.py"])
    root.destroy()

def window():
  root.destroy()
  
    
button1 = tk.Button(root, text="HOME", command=con, width=12, height=1,font=('times 15 bold underline'),bd=0, bg="#3CB371", fg="white")
button1.place(x=635, y=250)

button2 = tk.Button(root, text="LOGIN",command=log,width=12, height=1,font=('times 15 bold underline'), bd=0,bg="#3CB371", fg="white")
button2.place(x=635, y=350)

button4 = tk.Button(root, text="EXIT", command=window, width=12, height=1,font=('times 15 bold underline'),bd=0,bg="#FF4500", fg="white")
button4.place(x=635, y=450)

root.mainloop()