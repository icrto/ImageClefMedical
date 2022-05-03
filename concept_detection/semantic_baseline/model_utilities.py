# PyTorch Imports
import torch
from torchvision.models import densenet121, resnet18



# Function: Freeze the feature extractor of the backbone
def freeze_feature_extractor(model, name, freeze=True):

    # If freeze, we freeze the model
    if freeze:

        # We freeze the feature extractor
        for param in model.parameters():
            param.requires_grad = False
            # print(param.name, param.requires_grad)
        

        # Check the classifier by its name
        if name == "densenet121":
            model.classifier.requires_grad = True
            # print(model.classifier, model.classifier.requires_grad)
        
        elif name == "resnet18":
            model.fc.requires_grad = True
            # print(model.fc, model.fc.requires_grad)

    else:
        
        # We unfreeze the feature extractor
        for param in model.parameters():
            param.requires_grad = True
            # print(param.name, param.requires_grad)

    return



# Function: Unfreeze the feature extractor of the backbone 
def unfreeze_feature_extractor(model, name):
    
    # It's an alias for the previous function
    freeze_feature_extractor(model=model, name=name, freeze=False)

    return



# Run this code
if __name__ == "__main__":

    # DenseNet121
    model = densenet121(progress=True, pretrained=True)
    model.classifier = torch.nn.Linear(1024, 2)
    print("Freezing...")
    freeze_feature_extractor(model=model, name="densenet121")
    print("Freezed.")
    print("Unfreezing.")
    unfreeze_feature_extractor(model=model, name="densenet121")
    print("Unfreezed.")


    # ResNet18
    model = resnet18(progress=True, pretrained=True)
    model.fc = torch.nn.Linear(512, 2)
    print("Freezing...")
    freeze_feature_extractor(model=model, name="resnet18")
    print("Freezed.")
    print("Unfreezing.")
    unfreeze_feature_extractor(model=model, name="resnet18")
    print("Unfreezed.")
