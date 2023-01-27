
### 1. Imports and class names setup ###
import gradio as gr
import os
import torch
import torchvision.transforms as T

from model import create_effnet_b2
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ['pizza', 'steak', 'sushi']

### 2. Model and transforms preparation ###
test_tsfm = T.Compose([T.Resize((224,224)),
                        T.ToTensor(),
                       T.Normalize(mean=[0.485, 0.456, 0.406], # 3. A mean of [0.485, 0.456, 0.406] (across each colour channel)
                         std=[0.229, 0.224, 0.225]) # 4. A standard deviation of [0.229, 0.224, 0.225] (across each colour channel),
                       ])

# Create EffNetB2 Model
effnetb2, test_transform = create_effnet_b2(num_of_class=len(class_names), 
                            transform=test_tsfm,
                            seed=42)

# saved_path = 'demos\foodvision_mini\09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth'
saved_path = '07_effnetb2_data_50_percent_10_epochs.pth'

print('Loading Model State Dictionary')
# Load saved weights
effnetb2.load_state_dict(
                torch.load(f=saved_path,
                           map_location=torch.device('cpu'), # load to CPU
                          )
                        )

print('Model Loaded ...')
### 3. Predict function ###

# Create predict function
from typing import Tuple, Dict

def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    img = test_tsfm(img).unsqueeze(0)
    
    # Put model into evaluation mode and turn on inference mode
    effnetb2.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    
    # Return the prediction dictionary and prediction time 
    return pred_labels_and_probs, pred_time

### 4. Gradio App ###

# Create title, description and article strings
title= 'FoodVision Mini 🍕🥩🍣'
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created by Chukwuka [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."


# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.inputs.Image(type='pil'), # What are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label='Prediction time (s)')], # Our fn has two outputs, therefore we have two outputs
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article
                   )
# Launch the demo
print('Gradio Demo Launched')
demo.launch()

