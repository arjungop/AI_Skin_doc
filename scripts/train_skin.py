"""
Hierarchical Training with Auxiliary Data
- Main Data: Trains Category + Disease heads
- Auxiliary Data: Trains Category head only (Loss masked for Disease head)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import json
from pathlib import Path
from tqdm import tqdm

# Import model
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.ml.model import HierarchicalSkinClassifier

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Load Metadata
    data_dir = Path(args.data_dir)
    with open(data_dir / "hierarchy.json") as f:
        meta = json.load(f)
        
    categories = meta["categories"]
    folder_to_category = meta["folder_to_category"]
    is_auxiliary = meta["is_auxiliary"]
    
    # Load Dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    folder_classes = dataset.classes  # All folders (main + aux)
    
    # Pre-calculate labels
    folder_to_cat_idx = {
        cls: categories.index(folder_to_category[cls]) 
        for cls in folder_classes
    }
    
    # Aux flag per folder index
    is_aux_idx = [is_auxiliary[cls] for cls in folder_classes]

    # Map main folders to disease index (0-18), Aux folders get -1
    # We need a consistent list of MAIN diseases for the model output
    main_diseases = []
    for cat in categories:
        main_diseases.extend(meta["hierarchy"][cat])
        
    folder_to_disease_idx = {}
    for cls in folder_classes:
        if is_auxiliary[cls]:
            folder_to_disease_idx[cls] = -1 # Ignore
        else:
            folder_to_disease_idx[cls] = main_diseases.index(cls)

    print(f"Main Diseases ({len(main_diseases)}): {main_diseases}")
    print(f"Auxiliary Folders: {[f for f in folder_classes if is_auxiliary[f]]}")

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model
    model = HierarchicalSkinClassifier(
        num_categories=len(categories),
        num_diseases=len(main_diseases), # 19
        backbone=args.backbone
    ).to(device)
    
    # Weighted Loss (Category)
    # Calculate counts per category (including aux)
    cat_counts = [0] * len(categories)
    for cls in folder_classes:
        count = len(list((data_dir / cls).iterdir()))
        cat_idx = folder_to_cat_idx[cls]
        cat_counts[cat_idx] += count
        
    cat_weights = torch.FloatTensor([sum(cat_counts)/(len(categories)*c) for c in cat_counts]).to(device)
    category_criterion = nn.CrossEntropyLoss(weight=cat_weights, label_smoothing=0.1)
    
    # Standard Disease Loss (Only for main classes)
    disease_criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, folder_indices in pbar:
            imgs, folder_indices = imgs.to(device), folder_indices.to(device)
            
            # Convert folder indices to labels
            batch_folders = [folder_classes[i] for i in folder_indices]
            
            cat_labels = torch.tensor([folder_to_cat_idx[f] for f in batch_folders]).to(device)
            dis_labels = torch.tensor([folder_to_disease_idx[f] for f in batch_folders]).to(device)
            
            # Forward
            optimizer.zero_grad()
            cat_logits, dis_logits = model(imgs)
            
            # Loss Calculation
            loss_cat = category_criterion(cat_logits, cat_labels)
            
            # Disease loss (automatically ignores -1/aux labels)
            loss_dis = disease_criterion(dis_logits, dis_labels)
            
            # Combined Loss
            # If batch has ONLY aux data, dis_loss might be NaN/0 from ignore_index
            if torch.isnan(loss_dis): loss_dis = 0.0
            
            total_loss = 0.3 * loss_cat + 0.7 * loss_dis
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            pbar.set_postfix({'loss': f"{running_loss / (pbar.n + 1):.3f}"})
            
        # Validation
        model.eval()
        dis_correct = 0
        dis_total = 0
        
        with torch.no_grad():
            for imgs, folder_indices in val_loader:
                imgs, folder_indices = imgs.to(device), folder_indices.to(device)
                batch_folders = [folder_classes[i] for i in folder_indices]
                dis_labels = torch.tensor([folder_to_disease_idx[f] for f in batch_folders]).to(device)
                
                _, dis_logits = model(imgs)
                _, preds = dis_logits.max(1)
                
                # Only count main diseases for accuracy
                mask = dis_labels != -1
                if mask.sum() > 0:
                    dis_correct += preds[mask].eq(dis_labels[mask]).sum().item()
                    dis_total += mask.sum().item()
        
        val_acc = 100. * dis_correct / max(dis_total, 1)
        print(f"Val Disease Acc (Main Classes): {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = Path(args.output)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
            # Save minimal meta for inference
            meta_out = {
                "hierarchy": {"categories": categories}, # minimal
                "folder_classes": main_diseases, # For inference mapping
                "backbone": args.backbone
            }
            with open(str(save_path).replace('.pth', '.json'), 'w') as f:
                json.dump(meta_out, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', default='data/unified_train')
    parser.add_argument('--backbone', default='resnet18')
    parser.add_argument('--output', default='backend/ml/weights/hierarchical_skin.pth')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--img-size', type=int, default=224, help='Input image size')
    
    args = parser.parse_args()
    train(args)
