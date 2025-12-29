import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from model import NeuralNavigator
from data_loader import NavigationDataset

def predict(idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = NeuralNavigator()
    model.load_state_dict(torch.load("navigator_model.pth", map_location=device))
    model.eval()

    dataset = NavigationDataset('assignment_dataset', split='test_data')
    image, text, _ = dataset[idx]
    
    with torch.no_grad():
        path = model(image.unsqueeze(0), text.unsqueeze(0))
    
    # Visualization
    path = path.squeeze().numpy() * 128.0
    img = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    
    for i in range(len(path) - 1):
        cv2.line(img, (int(path[i][0]), int(path[i][1])), 
                      (int(path[i+1][0]), int(path[i+1][1])), (255, 0, 0), 2)
        cv2.circle(img, (int(path[i+1][0]), int(path[i+1][1])), 3, (0, 0, 255), -1)

    plt.imshow(img)
    plt.title("Predicted Path")
    plt.show()
    plt.imsave(f"result_{idx}.png", img)
    print(f"Result saved to result_{idx}.png")

if __name__ == "__main__":
    predict(0) # Change index to test different images