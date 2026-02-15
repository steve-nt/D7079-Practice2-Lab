"""
PART II: HRR Defense Implementation
Holographically Reduced Representations for Privacy Protection

This script implements the HRR defense mechanism from the paper:
"Crypto-Oriented Neural Architecture Design" (arXiv:2206.05893)

HRR uses 2D circular convolution in the frequency domain to "bind" inputs
with secret keys, making the model's intermediate representations uninformative
without the secret. This provides privacy against membership inference attacks.
"""

# Import necessary libraries
import numpy as np  # Numerical operations
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torchvision  # Computer vision datasets and models
import torchvision.transforms as transforms  # Data preprocessing
from torch.utils.data import DataLoader  # Data loading utilities
import torch.nn.functional as F  # Functional neural network operations
from torchvision.models import resnet18  # ResNet-18 architecture
import matplotlib.pyplot as plt  # Plotting library

# Device configuration - use GPU if available for faster computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =====================
# HRR Operations
# =====================

def generate_secret(H, W, C):
    """
    Generate a secret key for HRR binding operation
    
    The secret is a random tensor with unit magnitude in frequency domain.
    This ensures good binding properties and numerical stability.
    
    Process:
    1. Sample from normal distribution (Xavier-like initialization)
    2. Transform to frequency domain using 2D FFT
    3. Project to unit magnitude (normalize)
    4. Transform back to spatial domain
    
    Args:
        H: Height of image (e.g., 32 for CIFAR-10)
        W: Width of image (e.g., 32 for CIFAR-10)  
        C: Number of channels (e.g., 3 for RGB)
    
    Returns:
        Secret tensor of shape (C, H, W) - one secret per channel
    """
    # Sample from normal distribution with variance scaling
    # Factor 1/sqrt(H*W*C) helps maintain stable gradients
    s = torch.randn(C, H, W, device=device) * (1.0 / np.sqrt(H * W * C))
    
    # Transform to frequency domain using 2D Fast Fourier Transform
    # dim=(1, 2) applies FFT across height and width dimensions
    F_s = torch.fft.fft2(s, dim=(1, 2))
    
    # Compute magnitude (absolute value) in frequency domain
    magnitude = torch.abs(F_s)
    
    # Project to unit magnitude: divide by magnitude
    # Add epsilon (1e-10) to avoid division by zero
    # .real extracts real part after inverse FFT
    s_projected = torch.fft.ifft2(F_s / (magnitude + 1e-10), dim=(1, 2)).real
    
    return s_projected


def binding_2d(x, s):
    """
    Bind (obfuscate) image x with secret s using 2D circular convolution
    
    This is the core HRR operation. In the frequency domain, circular
    convolution becomes element-wise multiplication, making it efficient.
    
    Mathematical operation: x ⊛ s = F^(-1)[F(x) * F(s)]
    where F is 2D FFT, * is element-wise multiplication
    
    Uses 2D FFT for efficient computation (O(n log n) instead of O(n²))
    
    Args:
        x: Input image [C x H x W] - original data to protect
        s: Secret vector [C x H x W] - encryption key
    
    Returns:
        Bound (obfuscated) image [C x H x W] - can be sent to untrusted server
    """
    # Step 1: Transform input image to frequency domain
    # dim=(1, 2) means apply FFT across height and width, separately for each channel
    F_x = torch.fft.fft2(x, dim=(1, 2))
    
    # Step 2: Transform secret to frequency domain
    F_s = torch.fft.fft2(s, dim=(1, 2))
    
    # Step 3: Element-wise multiplication in frequency domain
    # This implements circular convolution (⊛) in spatial domain
    # Multiplication in frequency domain = convolution in spatial domain (convolution theorem)
    B = F_x * F_s
    
    # Step 4: Inverse FFT to get result back in spatial domain
    # .real extracts real part (imaginary part should be ~0 due to real inputs)
    bound = torch.fft.ifft2(B, dim=(1, 2)).real
    
    return bound


def unbinding_2d(B, s):
    """
    Unbind (decrypt) bound image B using the secret s
    
    This reverses the binding operation to recover the original image.
    Only possible with the correct secret key!
    
    Mathematical operation: B ⊛ s† = F^(-1)[F(B) * F(s)^†]
    where s† is the "inverse" of s (actually complex conjugate / magnitude²)
    
    Args:
        B: Bound (obfuscated) image [C x H x W] - received from server
        s: Secret vector [C x H x W] - decryption key (same as encryption key)
    
    Returns:
        Unbound (decrypted) image [C x H x W] - approximate reconstruction of original
    """
    # Step 1: Transform secret to frequency domain
    F_s = torch.fft.fft2(s, dim=(1, 2))
    
    # Step 2: Compute inverse secret s†
    # For complex numbers: inverse ≈ conjugate / (magnitude²)
    # torch.conj() computes complex conjugate
    # torch.abs(F_s)**2 computes magnitude squared
    # Add epsilon to avoid division by zero
    F_s_inv = torch.conj(F_s) / (torch.abs(F_s) ** 2 + 1e-10)
    
    # Apply unbinding
    F_B = torch.fft.fft2(B, dim=(1, 2))
    unbound = torch.fft.ifft2(F_B * F_s_inv, dim=(1, 2)).real
    
    return unbound


# =====================
# Network Architectures
# =====================

class ModifiedResNet18(nn.Module):
    """
    Modified ResNet-18 with encoder-decoder structure
    Input and output have same dimensions (required for HRR)
    """
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        
        # Encoder: Use ResNet-18 backbone
        resnet = resnet18(pretrained=False)
        
        # Remove the fully connected layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        # Decoder: Upsample back to original dimensions
        self.decoder = nn.Sequential(
            # From 512 to 256
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # From 256 to 128
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # From 128 to 64
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # From 64 to 3 (RGB)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final layer to match input channels
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
    
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Decoder
        x = self.decoder(x)
        
        return x


class PredictionNetwork(nn.Module):
    """
    Prediction network that takes unbound output and predicts class
    Similar to standard CNN classifier
    """
    def __init__(self, num_classes=10):
        super(PredictionNetwork, self).__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class GradientReverseLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training
    Forward: identity function
    Backward: negates gradients
    """
    @staticmethod
    def forward(ctx, x):
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class AdversarialNetwork(nn.Module):
    """
    Adversarial network that tries to classify without the secret
    Uses gradient reversal to force main network to be uninformative
    """
    def __init__(self, num_classes=10):
        super(AdversarialNetwork, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Apply gradient reversal
        x = GradientReverseLayer.apply(x)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# =====================
# Training Functions
# =====================

def train_hrr_model(trainloader, epochs=50, use_adversarial=True):
    """
    Train HRR-protected model using CSPS approach
    
    Args:
        trainloader: DataLoader for training data
        epochs: Number of training epochs
        use_adversarial: Whether to use adversarial network
    
    Returns:
        Tuple of (main_network, prediction_network, adversarial_network)
    """
    # Initialize networks
    main_network = ModifiedResNet18().to(device)
    pred_network = PredictionNetwork().to(device)
    adv_network = AdversarialNetwork().to(device) if use_adversarial else None
    
    # Optimizers
    optimizer_main = optim.Adam(main_network.parameters(), lr=0.001)
    optimizer_pred = optim.Adam(pred_network.parameters(), lr=0.001)
    optimizer_adv = optim.Adam(adv_network.parameters(), lr=0.001) if use_adversarial else None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        main_network.train()
        pred_network.train()
        if adv_network:
            adv_network.train()
        
        running_loss = 0.0
        correct_pred = 0
        correct_adv = 0
        total = 0
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Generate new secrets for each sample in batch
            C, H, W = images.shape[1], images.shape[2], images.shape[3]
            secrets = torch.stack([generate_secret(H, W, C) for _ in range(batch_size)])
            
            # Bind inputs
            bound_images = torch.stack([
                binding_2d(images[j], secrets[j]) for j in range(batch_size)
            ])
            
            # Forward through main network (would be on untrusted server)
            r = main_network(bound_images)
            
            # Prediction network (user side with secret)
            unbound = torch.stack([
                unbinding_2d(r[j], secrets[j]) for j in range(batch_size)
            ])
            pred_output = pred_network(unbound)
            
            # Calculate prediction loss
            loss_pred = criterion(pred_output, labels)
            
            # Adversarial network (tries without secret)
            if adv_network:
                adv_output = adv_network(r)
                loss_adv = criterion(adv_output, labels)
                total_loss = loss_pred + loss_adv
            else:
                total_loss = loss_pred
            
            # Backward pass
            optimizer_main.zero_grad()
            optimizer_pred.zero_grad()
            if optimizer_adv:
                optimizer_adv.zero_grad()
            
            total_loss.backward()
            
            optimizer_main.step()
            optimizer_pred.step()
            if optimizer_adv:
                optimizer_adv.step()
            
            # Statistics
            running_loss += total_loss.item()
            _, predicted = torch.max(pred_output.data, 1)
            total += labels.size(0)
            correct_pred += (predicted == labels).sum().item()
            
            if adv_network:
                _, predicted_adv = torch.max(adv_output.data, 1)
                correct_adv += (predicted_adv == labels).sum().item()
            
            if (i + 1) % 50 == 0:
                pred_acc = 100 * correct_pred / total
                adv_acc = 100 * correct_adv / total if adv_network else 0
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/50:.4f}, Pred Acc: {pred_acc:.2f}%, '
                      f'Adv Acc: {adv_acc:.2f}%')
                running_loss = 0.0
        
        epoch_pred_acc = 100 * correct_pred / total
        epoch_adv_acc = 100 * correct_adv / total if adv_network else 0
        print(f'Epoch [{epoch+1}/{epochs}] completed.')
        print(f'  Prediction Accuracy: {epoch_pred_acc:.2f}%')
        if adv_network:
            print(f'  Adversarial Accuracy: {epoch_adv_acc:.2f}% (should be low)')
    
    return main_network, pred_network, adv_network


def test_hrr_model(main_network, pred_network, testloader):
    """
    Test HRR-protected model
    
    Returns:
        Test accuracy
    """
    main_network.eval()
    pred_network.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            # Generate secrets
            C, H, W = images.shape[1], images.shape[2], images.shape[3]
            secrets = torch.stack([generate_secret(H, W, C) for _ in range(batch_size)])
            
            # Bind, process, unbind
            bound_images = torch.stack([
                binding_2d(images[j], secrets[j]) for j in range(batch_size)
            ])
            
            r = main_network(bound_images)
            
            unbound = torch.stack([
                unbinding_2d(r[j], secrets[j]) for j in range(batch_size)
            ])
            
            outputs = pred_network(unbound)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main():
    """Main execution function for Part II"""
    
    print("=" * 80)
    print("PART II: HRR DEFENSE IMPLEMENTATION")
    print("=" * 80)
    
    # Configuration
    TRAIN_EPOCHS = 30  # Can increase for better results
    BATCH_SIZE = 32  # Smaller batch size due to HRR overhead
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    print("\nLoading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # Use subset for faster training during development
    # Remove this for full training
    train_subset_size = 10000
    trainset = torch.utils.data.Subset(trainset, range(train_subset_size))
    
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Train HRR-protected model
    print("\n" + "=" * 80)
    print("Training HRR-Protected Model...")
    print("=" * 80)
    
    main_net, pred_net, adv_net = train_hrr_model(trainloader, epochs=TRAIN_EPOCHS,
                                                    use_adversarial=True)
    
    # Test model
    print("\n" + "=" * 80)
    print("Testing HRR-Protected Model...")
    print("=" * 80)
    
    test_acc = test_hrr_model(main_net, pred_net, testloader)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save models
    torch.save(main_net.state_dict(), 'hrr_main_network.pth')
    torch.save(pred_net.state_dict(), 'hrr_pred_network.pth')
    if adv_net:
        torch.save(adv_net.state_dict(), 'hrr_adv_network.pth')
    
    print("\nModels saved successfully!")
    
    # Train baseline model (no HRR) for comparison
    print("\n" + "=" * 80)
    print("Training Baseline Model (No HRR)...")
    print("=" * 80)
    
    baseline_model = resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)
    
    baseline_model.train()
    for epoch in range(TRAIN_EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = baseline_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{TRAIN_EPOCHS}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/50:.4f}, Acc: {100*correct/total:.2f}%')
                running_loss = 0.0
    
    # Test baseline
    baseline_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = baseline_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    baseline_acc = 100 * correct / total
    print(f"\nBaseline Test Accuracy: {baseline_acc:.2f}%")
    
    torch.save(baseline_model.state_dict(), 'baseline_model.pth')
    
    # Summary
    print("\n" + "=" * 80)
    print("PART II COMPLETE - Summary")
    print("=" * 80)
    print(f"HRR-Protected Model Accuracy: {test_acc:.2f}%")
    print(f"Baseline Model Accuracy: {baseline_acc:.2f}%")
    print(f"Accuracy Drop: {baseline_acc - test_acc:.2f}%")
    print("\nAll models have been saved.")
    print("Next: Run RMIA attack on both models to compare effectiveness.")
    print("=" * 80)


if __name__ == '__main__':
    main()
