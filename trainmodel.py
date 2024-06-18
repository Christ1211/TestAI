import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torch.optim import SGD
import yaml


# Load configuration from YAML file
with open('data/coco128.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)


# Initialize the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 91  # 91 classes in COCO dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Initialize COCO dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = CocoDetection(root="../../datasets/coco128/images/train2017", annFile="../../datasets/coco128/annotations/instances_train2017.json", transform=transform)

# Initialize DataLoader
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

# Define optimizer and loss function
optimizer = SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=0.0005)
loss_fn = torch.nn.CrossEntropyLoss()  # Check if this loss function is appropriate for your task

# Training loop
for epoch in range(config['num_epochs']):
    for images, targets in dataloader:
        optimizer.zero_grad()
        loss = 0
        for i in range(len(images)):
            output = model([images[i]])
            loss += loss_fn(output, targets[i]['labels'])  # Check if this loss calculation is appropriate
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {loss.item()}")

# Save trained model
torch.save(model.state_dict(), 'trained_model.pth')
