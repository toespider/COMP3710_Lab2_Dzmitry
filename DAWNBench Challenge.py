import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# ---------------------------------------------------------
# 1. CIFAR-10 Adapted ResNet-18 (Built from scratch)
# ---------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18CIFAR, self).__init__()
        self.in_planes = 64
        # Key change for CIFAR: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.linear(out)


# ---------------------------------------------------------
# 2. In-Memory GPU Batch Augmentation
# ---------------------------------------------------------
def get_gpu_data(device):
    # Standard normalization constants
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Convert directly to full tensors in GPU memory
    train_x = (torch.tensor(trainset.data).permute(0, 3, 1, 2).float() / 255.0).to(device)
    train_x = (train_x - mean) / std
    train_y = torch.tensor(trainset.targets, dtype=torch.long).to(device)

    test_x = (torch.tensor(testset.data).permute(0, 3, 1, 2).float() / 255.0).to(device)
    test_x = (test_x - mean) / std
    test_y = torch.tensor(testset.targets, dtype=torch.long).to(device)

    return (train_x, train_y), (test_x, test_y)


def augment_batch(images):
    # Random Crop with reflection padding
    padded = nn.functional.pad(images, (4, 4, 4, 4), mode='reflect')
    crop_x = torch.randint(0, 8, (1,)).item()
    crop_y = torch.randint(0, 8, (1,)).item()
    cropped = padded[:, :, crop_y:crop_y + 32, crop_x:crop_x + 32]

    # Random Horizontal Flip
    if torch.rand(1).item() > 0.5:
        cropped = torch.flip(cropped, dims=[3])
    return cropped


# ---------------------------------------------------------
# 3. Demonstration & Training Runner
# ---------------------------------------------------------
def train_and_eval():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    print(f"Using device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    (train_x, train_y), (test_x, test_y) = get_gpu_data(device)

    epochs = 30
    batch_size = 512
    n_samples = train_x.size(0)
    steps_per_epoch = (n_samples + batch_size - 1) // batch_size

    model = ResNet18CIFAR().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.4, total_steps=epochs * steps_per_epoch, pct_start=0.25, div_factor=10, final_div_factor=100
    )
    scaler = torch.amp.GradScaler('cuda')

    # Demonstration Requirement: Run single inference pass
    print("\n--- Running Single Inference Pass (Demonstration Req) ---")
    model.eval()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            _ = model(test_x[:16])
    print("Inference step complete.")

    # Demonstration Requirement: Run single training epoch
    print("--- Running Single Training Epoch (Demonstration Req) ---")
    model.train()
    t0 = time.time()
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i + batch_size]
        bx = augment_batch(train_x[idx])
        by = train_y[idx]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            preds = model(bx)
            loss = criterion(preds, by)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    print(f"Single epoch complete in {time.time() - t0:.2f} seconds.\n")

    # Full Fast Training to >94% Accuracy
    print(f"--- Starting Full Training for {epochs} Epochs ---")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            bx = augment_batch(train_x[idx])
            by = train_y[idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                preds = model(bx)
                loss = criterion(preds, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # Validation at key epochs
        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    test_preds = model(test_x)
                    acc = (test_preds.argmax(dim=1) == test_y).float().mean().item() * 100
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | Test Accuracy: {acc:.2f}% | Elapsed: {time.time() - start_time:.1f}s")

    total_time = time.time() - start_time
    print(f"\nFinal Test Accuracy: {acc:.2f}% achieved in {total_time:.2f} seconds.")


if __name__ == '__main__':
    train_and_eval()