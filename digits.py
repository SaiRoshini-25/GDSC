import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pygame
import numpy as np
import cv2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

def process_drawing(screen):
    surface = pygame.surfarray.array3d(screen)
    gray = cv2.cvtColor(surface, cv2.COLOR_RGB2GRAY)
    gray = np.transpose(gray, (1, 0))
    _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    gray = gray[y:y+h, x:x+w]
    gray = cv2.resize(gray, (20, 20))
    padded = np.zeros((28, 28), dtype=np.uint8)
    padded[4:24, 4:24] = gray
    padded = padded.astype(np.float32) / 255.0
    padded = (padded - 0.1307) / 0.3081
    tensor = torch.tensor(padded).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

def predict_digit(screen):
    image = process_drawing(screen)
    if image is None:
        return None
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    return prediction.item()

def draw_digit():
    pygame.init()
    size = 280
    screen = pygame.display.set_mode((size, size + 50))
    pygame.display.set_caption("Draw a Digit - Press C to clear")
    clock = pygame.time.Clock()
    screen.fill((0, 0, 0))
    drawing = False
    prediction = None
    font = pygame.font.Font(None, 36)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                prediction = predict_digit(screen)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                screen.fill((0, 0, 0))
                prediction = None
            if event.type == pygame.MOUSEMOTION and drawing:
                pygame.draw.circle(screen, (255, 255, 255), event.pos, 10)

        if prediction is not None:
            pygame.draw.rect(screen, (0, 0, 0), (0, size, size, 50))
            text = font.render(f"Prediction: {prediction}", True, (0, 255, 0))
            screen.blit(text, (10, size + 10))

        pygame.display.flip()
        clock.tick(60)

draw_digit()
