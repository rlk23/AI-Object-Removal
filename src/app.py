from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

def process_image(image_path):
    # Find DeepFill weights
    deepfill_weights_path = None
    for f in os.listdir('src/models'):
        if f.endswith('.pth'):
            deepfill_weights_path = os.path.join('src/models', f)
            break

    if deepfill_weights_path is None:
        raise FileNotFoundError("DeepFill weights file not found.")

    print("Creating rcnn model")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    transforms = weights.transforms()
    rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
    rcnn = rcnn.eval()

    print("Creating deepfill model")
    deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)

    model = ObjectRemove(segmentModel=rcnn,
                         rcnn_transforms=transforms,
                         inpaintModel=deepfill,
                         image_path=image_path)
    print("Running model")
    output = model.run()

    print("Saving result images")
    # Original image with bounding box
    img = cv2.cvtColor(model.image_orig[0].permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
    boxed = cv2.rectangle(img, (model.box[0], model.box[1]), (model.box[2], model.box[3]), (0, 255, 0), 2)
    boxed = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    boxed_path = os.path.join(app.config['RESULT_FOLDER'], 'boxed_image.png')
    plt.imsave(boxed_path, boxed.astype(np.float32) / 255.0)
    print("Boxed image saved to", boxed_path)

    # Masked image
    masked_path = os.path.join(app.config['RESULT_FOLDER'], 'masked_image.png')
    masked_image = model.image_masked.permute(1, 2, 0).detach().numpy()
    plt.imsave(masked_path, masked_image)
    print("Masked image saved to", masked_path)

    # Inpainted image
    inpainted_path = os.path.join(app.config['RESULT_FOLDER'], 'inpainted_image.png')
    plt.imsave(inpainted_path, output)
    print("Inpainted image saved to", inpainted_path)

    return boxed_path, masked_path, inpainted_path

@app.route('/')
def index():
    return '''
    <html>
    <body>
        <h1>Upload an Image</h1>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file">
            <input type="submit" value="Upload">
        </form>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print("File saved to", file_path)
        try:
            boxed_path, masked_path, inpainted_path = process_image(file_path)
            return jsonify({
                "boxed_image": boxed_path,
                "masked_image": masked_path,
                "inpainted_image": inpainted_path
            })
        except Exception as e:
            print("Error processing image:", e)
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
