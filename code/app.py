import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
from process import process_answer_sheet


def upload_file(my_w, result_view):
    global img, filename
    f_types = [('Jpg Files', '*.jpg'), ('Png Files', '*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=Image.open(filename)
    width, height = img.size
    width_new = int(width/3)
    height_new = int(height/3)
    img_resized = img.resize((width_new, height_new))
    img=ImageTk.PhotoImage(image=img_resized)
    b2 =tk.Button(my_w,image=img) # using Button 
    b2.grid(row=1,column=2)
    result_view.delete(1.0, tk.END)  
    
def display_result(result_view):
    if filename is None:
        return
    result_info, result_answer = process_answer_sheet(filename)
    result_view.configure(state='normal')
    result_view.insert(tk.END, "Info: \n")
    result_view.insert(tk.END, str(result_info))
    result_view.insert(tk.END, "\n\n\nAnswer: \n")
    print(type(result_answer))
    for key in result_answer.keys():
        result_view.insert(tk.END, f"Question: {int(key)} -> {result_answer[key]} \n") 
    result_view.configure(state='disable')

def run_app():
    my_w = tk.Tk()
    my_w.geometry("1600x900")  # Size of the window
    my_w.title('Multiple choice detection tool')
    result_view = tk.Text(my_w, height= 32, width=32)
    result_view.grid(row=1, column=3)
    # define button for processing
    upload_btn = tk.Button(my_w, text='Upload File', 
    width=20,command = lambda:upload_file(my_w, result_view))
    upload_btn.grid(row=1,column=1)
    detect_btn = tk.Button(my_w, text='Detect', 
    width=20,command = lambda:display_result(result_view))
    detect_btn.grid(row=2,column=3)
    
    my_w.mainloop()  # Keep the window open
    
if __name__ == '__main__':
    run_app()