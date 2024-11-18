import matplotlib
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import io
import base64
from flask import Flask, render_template
import threading
from model import MNISTNet
import random
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for storing training progress
training_losses = []
training_accuracies = []
validation_accuracies = []
current_epoch = 0
is_training = True
test_images = []
test_predictions = []
test_labels = []

def get_plot():
    # Check if we have any data to plot
    if not training_losses and not training_accuracies and not validation_accuracies:
        # Create empty plots with messages if no data
        plt.style.use('bmh')
        fig = plt.figure(figsize=(15, 5))
        
        for i in range(3):
            ax = plt.subplot(1, 3, i+1)
            ax.text(0.5, 0.5, 'Waiting for training data...', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout(pad=3.0)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    plt.style.use('bmh')
    fig = plt.figure(figsize=(15, 5))
    
    # Training loss plot
    ax1 = plt.subplot(1, 3, 1)
    if training_losses:
        ax1.plot(training_losses, color='#2ecc71', linewidth=2, label='Loss')
        ax1.fill_between(range(len(training_losses)), training_losses, alpha=0.1, color='#2ecc71')
    ax1.set_title('Training Loss', pad=15, fontsize=12, fontweight='bold')
    ax1.set_xlabel('Batch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(frameon=True, facecolor='white', framealpha=1)
    
    # Training accuracy plot
    ax2 = plt.subplot(1, 3, 2)
    if training_accuracies:
        ax2.plot(training_accuracies, color='#3498db', linewidth=2, label='Accuracy')
        ax2.fill_between(range(len(training_accuracies)), training_accuracies, alpha=0.1, color='#3498db')
    ax2.set_title('Training Accuracy', pad=15, fontsize=12, fontweight='bold')
    ax2.set_xlabel('Batch', fontsize=10)
    ax2.set_ylabel('Accuracy', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax2.legend(frameon=True, facecolor='white', framealpha=1)
    
    # Validation accuracy plot
    ax3 = plt.subplot(1, 3, 3)
    if validation_accuracies:
        ax3.plot(validation_accuracies, color='#e74c3c', linewidth=2, marker='o', 
                label='Val Accuracy', markersize=8)
        ax3.fill_between(range(len(validation_accuracies)), validation_accuracies, 
                        alpha=0.1, color='#e74c3c')
    ax3.set_title('Validation Accuracy', pad=15, fontsize=12, fontweight='bold')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Accuracy', fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax3.legend(frameon=True, facecolor='white', framealpha=1)
    
    # Improve the appearance
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(pad=3.0)
    
    # Save plot with white background
    fig.patch.set_facecolor('white')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def home():
    try:
        # Get the latest metrics safely
        latest_loss = training_losses[-1] if training_losses else 0
        latest_train_acc = training_accuracies[-1] if training_accuracies else 0
        latest_val_acc = validation_accuracies[-1] if validation_accuracies else 0
        
        return render_template('index.html', 
                             plot=get_plot(), 
                             epoch=current_epoch,
                             is_training=is_training,
                             test_images=test_images,
                             test_predictions=test_predictions,
                             test_labels=test_labels,
                             latest_loss=latest_loss,
                             latest_train_acc=latest_train_acc,
                             latest_val_acc=latest_val_acc)
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}")
        return f"An error occurred: {str(e)}", 500

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    return correct / total

def train_model():
    global current_epoch, is_training, test_images, test_predictions, test_labels
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Data loading
    logger.info("Loading datasets...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Split training data into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Model setup
    logger.info("Initializing model...")
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    logger.info("Model initialized successfully")

    # Training
    best_val_accuracy = 0.0
    num_epochs = 10
    logger.info("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        current_epoch = epoch + 1
        
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)
            
            # Update running statistics
            running_loss += loss.item()
            running_correct += correct
            total_samples += len(data)
            
            # Store metrics
            training_losses.append(loss.item())
            training_accuracies.append(accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{accuracy:.4f}'
            })
        
        # Evaluate on validation set
        val_accuracy = evaluate_model(model, val_loader, device)
        validation_accuracies.append(val_accuracy)
        
        # Log epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = running_correct / total_samples
        logger.info(
            f'Epoch {epoch+1}/{num_epochs} - '
            f'Loss: {epoch_loss:.4f} - '
            f'Training Accuracy: {epoch_accuracy:.4f} - '
            f'Validation Accuracy: {val_accuracy:.4f}'
        )
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f'New best validation accuracy: {val_accuracy:.4f}')

    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy = evaluate_model(model, test_loader, device)
    logger.info(f"Final test accuracy: {test_accuracy:.4f}")

    # Testing and visualization
    logger.info("Starting model evaluation...")
    is_training = False
    model.eval()
    
    # Get random test samples
    data_iterator = iter(test_loader)
    data, labels = next(data_iterator)
    
    # Select 10 random indices
    random_indices = random.sample(range(len(data)), 10)
    
    logger.info("Generating test predictions for visualization...")
    with torch.no_grad():
        for idx in random_indices:
            img = data[idx].to(device)
            label = labels[idx].item()
            
            output = model(img.unsqueeze(0))
            pred = output.argmax(dim=1, keepdim=True).item()
            
            # Convert image to base64 for display
            img_np = data[idx].numpy().squeeze()
            plt.figure(figsize=(2, 2))
            plt.imshow(img_np, cmap='gray')
            plt.axis('off')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            buf.seek(0)
            plt.close()
            
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            test_images.append(img_base64)
            test_predictions.append(pred)
            test_labels.append(label)
            
            logger.info(f'Test image {len(test_images)}/10 - '
                       f'Predicted: {pred}, Actual: {label}')

    logger.info("Evaluation completed!")

if __name__ == '__main__':
    # Start training in a separate thread
    logger.info("Starting training thread...")
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    
    # Start Flask server
    logger.info("Starting Flask server...")
    app.run(debug=False) 