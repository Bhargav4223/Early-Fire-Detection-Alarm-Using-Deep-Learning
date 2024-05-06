from PIL import Image
import sys
print(sys.executable)
from ultralytics import YOLO

# Load a pretrained YOLOv3 model
model = YOLO('yolov3.weights')

# Run inference on your images
results = model(['image1.jpg', 'image2.jpg'])

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    r.save(filename=f'results{i}.jpg')
