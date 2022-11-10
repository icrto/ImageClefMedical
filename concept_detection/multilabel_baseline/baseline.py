import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
from tqdm import tqdm
from asl import AsymmetricLoss

def build_model(pretrained, freeze_fe, nr_concepts):
    weights = None
    if pretrained:
        weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)

    if freeze_fe:
        for param in model.parameters():
            param.requires_grad = False

    # make the classification layer learnable
    if nr_concepts == 'all': nr_concepts = 8374
    else: nr_concepts = int(nr_concepts)

    model.classifier = nn.Linear(1024, nr_concepts)
    return model


# training function
def do_epoch(model, dataloader, criterion, device, optimizer=None, weights=None, validation=False):
    if validation == True: model.eval()
    else: model.train()
    counter = 0
    running_loss = 0.0
    with torch.set_grad_enabled(validation == False):
        for data in tqdm(dataloader):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            if validation == False: optimizer.zero_grad()
            # forward pass
            outputs = model(data)
            if not isinstance(criterion, AsymmetricLoss) and not isinstance(criterion, nn.BCEWithLogitsLoss):
                # apply sigmoid activation to get all the outputs between 0 and 1
                outputs = torch.sigmoid(outputs)
            # compute loss
            loss = criterion(outputs, target)
            if isinstance(criterion, nn.BCELoss) and weights:
                loss = (loss * weights).mean()
            # compute gradients
            if validation == False: 
                loss.backward()
                optimizer.step()
            running_loss += loss.item()

        return running_loss / counter