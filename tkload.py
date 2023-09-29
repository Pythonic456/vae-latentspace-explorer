import os, time, math, random
import torch
from main import VAE, ImageDataset
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tkinter as tk
from tkinter import Scale, Button, Canvas, Label

# Parameters
pow = 9
latent_dim = 30
image_size = (2**pow, 2**pow)
model_load_path = "vae_model.pt"  # Path to the pre-trained model
DEBUG = False  # Set to True to print latent space vectors and filenames

# Load pre-trained VAE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE.load(model_load_path, latent_dim, image_size, num_channels=3).to(device)  # Specify num_channels as 3 for RGB images

# Create a tkinter window for real-time image generation
root = tk.Tk()
root.title("VAE Latent Space Explorer")

# Initialize sliders for latent space variables with a horizontal orientation
latent_ss_frame = tk.Frame(root)
latent_space_sliders = [Scale(latent_ss_frame, from_=-4, to=4, resolution=0.01, orient='horizontal') for _ in range(latent_dim)]  # Adjust from_ and to values
latent_ss_frame.pack(expand=True, fill="both")

tlabel = Label(root, text='0.0ms')
tlabel.pack()

# Canvas to display the generated image
canvas = Canvas(root)
canvas.pack(expand=True, fill="both")  # Make canvas expand to fill available space

# Function to update the generated image when sliders are adjusted
def update_image(event=None, ret=False):
    s_time = time.time()
    with torch.no_grad():
        latent_space = [slider.get() for slider in latent_space_sliders]
        z = torch.tensor(latent_space, dtype=torch.float32).unsqueeze(0).to(device)
        generated_image = model.decode(z).squeeze().cpu().numpy()
        generated_image = np.transpose(generated_image, (1, 2, 0))  # Transpose to (H, W, C) format
        pil_image = Image.fromarray((generated_image * 255).astype('uint8'), 'RGB')  # Specify 'RGB' mode
        pil_image_res = pil_image.resize((200, 200), Image.NEAREST)
        tk_image = ImageTk.PhotoImage(image=pil_image_res)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_image, tags="image")
        canvas.tag_lower("image")
        canvas.image = tk_image
    tlabel.config(text=str(round((time.time()-s_time)*1000, 3))+'ms')
    if ret:
        return pil_image

def dorandom():
    for slider in latent_space_sliders:
        slider.set(random.uniform(-4, 4))  # Adjust the range to match the latent space range
    update_image()

def save_image():
    im = update_image(ret=True)
    im.save('output.png')

button_randomize = Button(latent_ss_frame, text='Random', command=dorandom)
button_randomize.grid(row=0, column=0)

button_save = Button(latent_ss_frame, text='Save', command=save_image)
button_save.grid(row=0, column=1)

r = 1
c = 0
# Pack the latent space sliders and bind them to the update_image function
for i, slider in enumerate(latent_space_sliders):
    slider.set(0.0)  # Set initial values to 0.0
    slider.grid(row=r, column=c)
    slider.config(command=update_image)
    c += 1
    if c >= math.sqrt(len(latent_space_sliders)):
        c = 0
        r += 1

# Button to quit the application
# quit_button = Button(root, text="Quit", command=root.destroy)
# quit_button.pack()

root.after(10, update_image)
print()
# Run the tkinter main loop
root.mainloop()
