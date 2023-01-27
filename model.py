
import torch
import torch.nn as nn
import torchvision


# Create an EffNetB2 feature extractor
def create_effnet_b2(num_of_class: str=3,
                     transform: torchvision.transforms=None,
                     seed=42
                     ):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model. 
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    
    # 1. Get the base mdoel with pretrained weights and send to target device
    model = torchvision.models.efficientnet_b2(pretrained=True)
    
    # 2. Freeze the base model layers
    for param in model.parameters():
        param.requires_grad = False
    
    # 3. Set the seeds    
    torch.manual_seed(seed)
    
    # 4. Change the classifier head
    model.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True),
                                     nn.Linear(1408, num_of_class, bias=True)
                                    )
    
    return model, transform

# mymodel = create_effnet_b2(num_of_class=3, 
#                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
#                            seed=42)
# print(mymodel)
