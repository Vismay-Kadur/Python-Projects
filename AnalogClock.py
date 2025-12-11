import tkinter as tk


def button_click():
    label.config(text="Button Clicked!")


# Create the main window
root = tk.Tk()
root.title("My First GUI")
root.geometry("300x200")

# Create a label
label = tk.Label(root, text="Hello, GUI!")
label.pack(pady=20)

# Create a button
button = tk.Button(root, text="Click Me", command=button_click)
button.pack()

# Start the event loop
root.mainloop()
