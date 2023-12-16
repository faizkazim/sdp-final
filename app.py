import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import subprocess
from ttkthemes import ThemedStyle

def run_script():
    try:
        # Replace 'your_script.py' with the actual name of your Python script
        subprocess.run(["python", "demo.py"])
    except Exception as e:
        tk.messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main GUI window
root = tk.Tk()
root.title("Dark-themed GUI Example")

# Apply a dark theme
style = ThemedStyle(root)
style.set_theme("equilux")

# Frame with dark background for the title
title_frame = ttk.Frame(root, style="TFrame")
title_frame.pack(fill=tk.BOTH, expand=True)

# Title label with white text
title_label = ttk.Label(title_frame, text="My Dark GUI", style="Title.TLabel")
title_label.pack(pady=20)

# Button to run the script with a white background
run_script_button = ttk.Button(root, text="Run Script", command=run_script, style="White.TButton")
run_script_button.pack(pady=20)

# Function to handle GUI window closing
def on_closing():
    if tk.messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

# Bind the closing event to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

# Configure styles
style.configure("TFrame", background="#363636")  # Dark background for the frame
style.configure("Title.TLabel", foreground="white", background="#363636", font=("Helvetica", 16, "bold"))  # White text for the title
style.configure("White.TButton", background="white", font=("Helvetica", 12))  # White button

# Add hover effect to the button
style.map("White.TButton",
          background=[("active", "white"), ("hover", "#c0c0c0")],
          foreground=[("active", "black"), ("hover", "black")])

# Start the GUI main loop
root.mainloop()
