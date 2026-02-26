import torch
from tqdm import tqdm
from helper_utils import show_model_layers
import mlflow

import mlflow
import mlflow.pytorch
from tqdm.notebook import tqdm
import torch


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs=10, device='mps', 
                experiment_name="reaction_transformer", run_name="baseline"):
    """
    Simple training function with MLflow tracking
    """
    # Start MLflow run
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        
        # Log hyperparameters (model config, batch size, etc.)
        mlflow.log_params({
            "num_epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "optimizer": optimizer.__class__.__name__,
            "initial_lr": optimizer.param_groups[0]['lr'],
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "device": str(device),
        })
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_total_loss = 0
            train_batches = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for src_batch, tgt_batch in train_bar:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                
                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                
                optimizer.zero_grad()
                outputs = model(src_batch, tgt_input)
                outputs = outputs.reshape(-1, outputs.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                loss = criterion(outputs, tgt_output)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_total_loss += loss.item()
                train_batches += 1
                
                avg_loss = train_total_loss / train_batches
                train_bar.set_postfix({'loss': f'{loss.item():.3f} (avg: {avg_loss:.3f})'})
            
            # Validation phase
            model.eval()
            val_total_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for src_batch, tgt_batch in val_bar:
                    src_batch = src_batch.to(device)
                    tgt_batch = tgt_batch.to(device)
                    
                    tgt_input = tgt_batch[:, :-1]
                    tgt_output = tgt_batch[:, 1:]
                    
                    outputs = model(src_batch, tgt_input)
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    tgt_output = tgt_output.reshape(-1)
                    
                    loss = criterion(outputs, tgt_output)
                    
                    val_total_loss += loss.item()
                    val_batches += 1
                    
                    val_bar.set_postfix({'loss': f'{loss.item():.3f}'})
            
            # Calculate average losses
            train_loss = train_total_loss / train_batches
            val_loss = val_total_loss / val_batches

            # Step scheduler
            scheduler.step()

            # Log LR
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Store history
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                }
                torch.save(checkpoint, "train_log/best_checkpoint.pt")
                mlflow.log_artifact("train_log/best_checkpoint.pt")
            
            # ========== MLFLOW LOGGING ==========
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("epoch", epoch, step=epoch)
            
            # Log loss ratio (useful for monitoring overfitting)
            mlflow.log_metric("loss_ratio", val_loss/train_loss, step=epoch)
            
            # Print epoch summary
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}\n')
        
        print(f"✅ MLflow run logged: {mlflow.active_run().info.run_id}")
    
    return model
