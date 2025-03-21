# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Include the problem statement and Dataset
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
</br>

### STEP 2:
</br>

### STEP 3:

Write your own steps
<br/>

## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models import VGG19_Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes

num_classes=len(train_dataset.classes)
in_features=model.classifier[-1].in_features
model.classifier[-1]=nn.Linear(in_features,num_classes)


# Include the Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# Train the Model
def train_model(model, train_loader,test_loader,num_epochs=30):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: K SANTHAN KUMAR")
    print("Register Number: 212223240065")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
Include your plot here

![image](https://github.com/user-attachments/assets/84da6d9d-a74e-4849-bf56-c8bcd2d7ed67)

</br>
</br>
</br>

### Confusion Matrix
Include confusion matrix here
</br>
![image](https://github.com/user-attachments/assets/3c02b6b4-dcc2-485b-b7ac-698323ca2085)

</br>
</br>

### Classification Report
Include Classification Report here
</br>
![image](https://github.com/user-attachments/assets/c37faed5-c026-45ba-b82d-8237c1fd9399)

</br>
</br>

### New Sample Prediction
</br>

![image](https://github.com/user-attachments/assets/edc69ba5-957f-45d3-94cb-1812c4b1013f)

</br>
</br>

## RESULT

Successfully executed the program for Implementation of transfer learning.
