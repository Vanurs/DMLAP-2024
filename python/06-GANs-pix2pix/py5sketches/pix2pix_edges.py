import torch
import numpy as np
from py5canvas import canvas
import cv2
from skimage import feature, io
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu' # Can try 'mps' if using mac

def load_model(model_path):
    return torch.jit.load(os.path.expanduser(model_path), map_location=device)

# Assume models are in parent directory
#model_path = '../models/edge2comics_11_generator.pt'
model_path = '../models/edge2comics_60_generator.pt'

generator = load_model(model_path)
# Do not 'eval', which will usually "freeze" batch normalization and dropout layers  that would block meaning that we still use the dropout layers
# https://arxiv.org/pdf/1607.08022.pdf
#generator.eval()

vin = canvas.VideoInput(size=(256, 256))
img = io.imread('../images/spock.jpg')

def generate_image(model, img):
    img = torch.tensor((img/255)*2.0 - 1.0, 
                       dtype=torch.float32).to(device) # Scale to -1, 1
    # Convert to torch batch form (NCHW)
    img = img[:, :, :, np.newaxis]
    img = img.permute(3, 2, 0, 1)
    res = model(img)[0].permute(1, 2, 0)
    return (res*0.5 + 0.5).detach().cpu().numpy()

def canny(img, sigma=1.0):
    grayimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grayimg = cv2.resize(grayimg, (256, 256))/255
    edges = (feature.canny(grayimg, sigma=sigma)*255).astype(np.uint8)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

def setup():
    create_canvas(1024, 512)
    
def draw():
    background(255)
    torch.manual_seed(223)
    np.random.seed(122)
    im =  vin.read()
    
    # Make sure image is 8 bit RGB
    im = im.astype(np.uint8)
    # Globally apply scale to fit window
    scale(height/256)

    edges = canny(im)
    # Blend input and processed images together and draw in left half
    image(np.clip((im.astype(float) + edges)/255, 0, 1))
    # Compute result and draw on right half
    res = generate_image(generator, edges)
    c.image(res, 256, 0)

if __name__== '__main__':
    # This gets automatically called when running with python rather than py5sketch
    import py5canvas
    py5canvas.run()
