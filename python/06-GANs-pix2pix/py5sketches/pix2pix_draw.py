import torch
import numpy as np
from py5canvas import canvas
import cv2
from skimage import feature, io
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def load_model(model_path):
    return torch.jit.load(os.path.expanduser(model_path), map_location=device)

# Assume models are in parent directory
model_path = '../models/landmarks2rembrandt_195_generator.pt'
#model_path = '../models/edge2comics_60_generator.pt'

generator = load_model(model_path)
# Do not 'eval', which will usually "freeze" batch normalization and dropout layers  that would block meaning that we still use the dropout layers
# https://arxiv.org/pdf/1607.08022.pdf
#generator.eval()

# Empty result initially
result = np.ones((256, 256, 3))

# Setting this to True will draw in realtime but might be slooow
realtime = False

def generate_image(model, img):
    img = torch.tensor((img/255)*2.0 - 1.0, 
                       dtype=torch.float32).to(device) # Scale to -1, 1
    # Convert to torch batch form (NCHW)
    img = img[:, :, :, np.newaxis]
    img = img.permute(3, 2, 0, 1)
    res = model(img)[0].permute(1, 2, 0)
    return (res*0.5 + 0.5).detach().cpu().numpy()

def setup():
    sketch.create_canvas(800, 400)
    #sketch.frame_rate(20) # Our framerate will be pretty low anyhow
    sketch.canvas.background(0) # Clear once 

def draw():
    stroke(255)
    no_fill()
    stroke_weight(2.0)
    # Draw in the left side of the window 
    if sketch.mouse_pressed:
        line(sketch.mouse_pos - sketch.mouse_delta, sketch.mouse_pos)
    if realtime:
        global result # We are modifying the global result variable from here
        img = get_image()[:,:width//2,:]  # Slice the left half of the image grabbed from the canvas
        img = cv2.resize(img, (256, 256)) # Resize it 
        result = generate_image(generator, img) # Generate result

    # Draw result on the right side
    image(result, [width/2, 0], [width/2, width/2])

def key_pressed(k, modifier):
    print('key pressed')
    
    c = chr(k) # Get key code pressed
    if c=='c': 
        # c Clears background
        background(0)
    if c==' ':
        # Space generates the image
        global result # We are modifying the global result variable from here
        img = get_image()[:,:width//2,:]  # Slice the left half of the image grabbed from the canvas
        img = cv2.resize(img, (256, 256)) # Resize it 
        result = generate_image(generator, img) # Generate result

if __name__== '__main__':
    # This gets automatically called when running with python rather than py5sketch
    import py5canvas
    py5canvas.run()
