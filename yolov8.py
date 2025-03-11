import os
import time
import torch
import psutil
from roboflow import Roboflow
from ultralytics import YOLO
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

class TrainingMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.update_interval = 60  # Update every minute
        self.metrics = {
            'epoch': [],
            'time': [],
            'cpu_usage': [],
            'memory_usage': [],
        }

    def format_time(self, seconds):
        return str(timedelta(seconds=int(seconds)))

    def update(self, epoch, total_epochs):
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            elapsed_time = current_time - self.start_time
            time_per_epoch = elapsed_time / (epoch + 1)
            remaining_epochs = total_epochs - (epoch + 1)
            estimated_remaining = time_per_epoch * remaining_epochs
            
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            self.metrics['epoch'].append(epoch + 1)
            self.metrics['time'].append(elapsed_time)
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory)
            
           
            
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                print(f"GPU Memory Usage: {gpu_memory:.2f} MB")
            
            print(f"{'='*50}\n")
            
            self.last_update = current_time

    def plot_metrics(self):
        """Plot training metrics after completion"""
        metrics_df = pd.DataFrame(self.metrics)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(metrics_df['epoch'], metrics_df['cpu_usage'])
        plt.title('CPU Usage Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('CPU Usage (%)')
        
        # Memory Usage
        plt.subplot(1, 2, 2)
        plt.plot(metrics_df['epoch'], metrics_df['memory_usage'])
        plt.title('Memory Usage Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Memory Usage (MB)')
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()

def setup_training():
    
    print("\nInitializing Training Environment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU tingzzzzz are detected type: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version is hereeeeee yuuu : {torch.version.cuda}")
        
        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"THE TOTAL GPU MEMORY: {gpu_memory:.2f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU I CANT AFFORD ONE")
    
    # System information
    print(f"\nSystem Information:")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"Available RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    
    return device

def main():
    # Dataset paths
    dataset_path = "C://Users//brind/Downloads//RenewedResearch//Brain-Tumor-2"
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    
    device = setup_training()
    
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("\nDownloading dataset...")
        rf = Roboflow(api_key="lJNDnTVskAnXQXadHX2f")
        project = rf.workspace("mike-cox-szip2").project("brain-tumor-omjcy")
        version = project.version(2)
        dataset = version.download("yolov8")
        print("Dataset downloaded successfully!!!!!!!!!!!!!!!!!!!!!! wo hoooooo")
    else:
        print(f"\nUsing existing dataset at {dataset_path}")
    
    # Initialize model
    print("\nInitializing YOLOv8 model...")
    model = YOLO('yolov8m-seg.pt')
    
    # Training parameters
    epochs = 75
    imgsz = 640
    batch_size = 8 if torch.cuda.is_available() else 4
    
    monitor = TrainingMonitor()
    
    def on_train_epoch_end(trainer):
        monitor.update(trainer.epoch, epochs)
    
    print("\nStarting training...")
    print(f"Training for {epochs} epochs")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch_size}")
    
    try:
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=4,
            
            patience=15,  
            save_period=5,  
            verbose=True
        )
        
        print("\nTraining completed successfully!")
        
        # Plot and save training metrics
        monitor.plot_metrics()
        print("\nTraining metrics have been saved to 'training_metrics.png'")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        
    finally:
        # Print final statistics
        total_time = time.time() - monitor.start_time
        print(f"\nTotal training time: {monitor.format_time(total_time)}")
        print(f"Average time per epoch: {monitor.format_time(total_time/epochs)}")
        
        # Save final model
        model.save('final_modelz.pt')
        print("\nModel saved as 'final_model.pt'")

if __name__ == "__main__":
    main()