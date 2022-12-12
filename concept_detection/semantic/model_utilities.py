# Function: Freeze the feature extractor of the backbone
def freeze_feature_extractor(model, name, freeze=True):

    # If freeze, we freeze the model
    if freeze:

        # We freeze the feature extractor
        for param in model.parameters():
            param.requires_grad = False

        # Check the classifier by its name
        if name.lower() == "densenet121".lower():
            for param in model.classifier.parameters():
                param.requires_grad = True

        elif name.lower() == "resnet18".lower():
            for param in model.fc.parameters():
                param.requires_grad = True

    else:

        # We unfreeze the feature extractor
        for param in model.parameters():
            param.requires_grad = True

    return


# Function: Unfreeze the feature extractor of the backbone
def unfreeze_feature_extractor(model, name):

    # It's an alias for the previous function
    freeze_feature_extractor(model=model, name=name, freeze=False)

    return
