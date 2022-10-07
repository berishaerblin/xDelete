import tkinter as tk
from tkinter import filedialog
from check import Predict as predict, Train as train

window = tk.Tk()
window.geometry("500x200")
predict_path = tk.StringVar()
train_path = tk.StringVar()


def train_directory():
    train_dir = filedialog.askdirectory()
    train_path.set(train_dir)


def train_model():
    train(train_path.get()).run()


def browse_func():
    file_name = filedialog.askopenfilename()
    predict_path.set(file_name)


def predict_picture():
    predict(predict_path.get()).run()


def show_window():
    train_browse_button = tk.Button(window, text="Browse train pictures", command=train_directory)
    run_training = tk.Button(window, text="Run Training", command=train_model)
    browse_button = tk.Button(window, text="Browse pictures to find & delete", command=browse_func)
    run_button = tk.Button(window, text="Run", command=predict_picture)
    train_browse_button.pack()
    run_training.pack()
    browse_button.pack()
    run_button.pack()
    window.mainloop()


show_window()
