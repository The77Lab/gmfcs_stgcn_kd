
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DataSet_Classification
import math
import os
import sys
# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from models.heads import Classifier
from models.stgcn import STGCN_2

from student_network import Small_STGCN


class STGCN_Classifier(nn.Module):

    def __init__(self,
                 backbone,
                 fc_latent_dim=512,
                 in_channels=256,
                 compressed_T_dim=31,
                 num_classes=0):
        super(STGCN_Classifier, self).__init__()

        args = backbone.copy()
        args.pop('type')
        self.backbone = STGCN_2(**args)
        self.cls_head = Classifier(
            num_classes=num_classes, dropout=0.5, latent_dim=fc_latent_dim, in_channels=in_channels, compressed_T_dim=compressed_T_dim)

    def forward(self, keypoint):
        """Define the computation performed at every call."""
        stage_1_out, stage_4_out, stage_7out, stage_10_out, x = self.backbone(keypoint)
        cls_score = self.cls_head(x)
        return stage_1_out, stage_4_out, stage_7out, stage_10_out, cls_score

batch_size = 128
sample_folder = "path to the data"
train_dataset = DataSet_Classification(os.path.join(script_dir, 'train_dataset14.npy'),
                        sample_folder, data_augmentation=False)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = DataSet_Classification(os.path.join(script_dir, 'val_dataset14.npy'),
                      sample_folder, data_augmentation=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

backbone_cfg = {
    'type': 'STGCN',
    'gcn_adaptive': 'init',
    'gcn_with_res': True,
    'tcn_type': 'mstcn',
    'graph_cfg': {
        'layout': 'coco',
        'mode': 'spatial'
    },
    'pretrained': None
}
# head_cfg = {'type': 'GCNHead', 'num_classes': 4, 'in_channels': 256}
model = STGCN_Classifier(backbone=backbone_cfg, num_classes=4)

# model.init_weights()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda:0'


# Load pre-trained weights to the backbone
backbone_state_dict = os.path.join(script_dir, 'model_weights.pth')
# load_checkpoint(model.backbone, backbone_state_dict)
tmp = torch.load(backbone_state_dict)

model.load_state_dict(tmp, strict=True)

# print(model)

# for param in model.backbone.parameters():
#     param.requires_grad = False

model = model.to(device)

model.eval()

graph_cfg = {
    'layout': 'coco',
    'mode': 'spatial'
}


def distill_loss(outputs, teacher_labels, T):
    outputs = F.softmax(outputs / T, dim=1)
    teacher_labels = F.softmax(teacher_labels / T, dim=1)
    loss = -torch.mean(torch.sum(teacher_labels * torch.log(outputs), dim=1))
    return loss

def distill_L2_loss(outputs, teacher_labels):
    loss = torch.mean((outputs - teacher_labels)**2)
    return loss

def middle_L2_loss(student_middle_layer_out, teacher_middle_layer_out):
    # print(student_middle_layer_out.shape, teacher_middle_layer_out.shape)
    loss = torch.mean((student_middle_layer_out - teacher_middle_layer_out)**2)
    # print(loss)
    return loss

# T = [1, 3, 5, 10]
# alphas = [0.1, 0.5, 1, 5, 10]
alpha = 1
block_cfg = 1
linear_cfg = 0
student_model = Small_STGCN(graph_cfg=graph_cfg, block_config=block_cfg, linear_config=linear_cfg)
student_model = student_model.to(device)

num_epochs = 10

hard_loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(student_model.parameters(), lr=0.0001, weight_decay=0.00005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

out_model_path = os.path.join('output_path' + str(block_cfg) + '_' + str(linear_cfg) + '.pth')
print(out_model_path)

val_best_acc = -math.inf
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    student_model.train()  # set the model to train mode
    epoch_pbar = tqdm(desc=f"Epoch {epoch+1}/{num_epochs}",
                    total=len(train_dataloader.dataset) / batch_size, position=0)

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_distill_loss = 0.0
    epoch_cross_entropy_loss = 0.0
    epoch_middle_0_loss = 0.0
    epoch_middle_1_loss = 0.0
    epoch_middle_2_loss = 0.0
    epoch_middle_3_loss = 0.0

    sample_count = 0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        if block_cfg == 0:
            student_middle_0, student_middle_1, student_middle_2, student_middle_3, outputs = student_model(inputs)
        elif block_cfg == 1:
            student_middle_1, student_middle_2, student_middle_3, outputs = student_model(inputs)
        elif block_cfg == 2:
            student_middle_2, student_middle_3, outputs = student_model(inputs)
        elif block_cfg == 3:
            student_middle_3, outputs = student_model(inputs)

        # teacher model forward pass
        with torch.no_grad():
            teacher_middle_0, teacher_middle_1, teacher_middle_2, teacher_middle_3, teacher_outputs = model(inputs)

        # dist_loss = distill_loss(outputs, teacher_outputs, temperature)
        dist_loss = distill_L2_loss(outputs, teacher_outputs)
        cross_entropy_loss = hard_loss(outputs, labels)
        if block_cfg == 0:
            middle_0_loss = middle_L2_loss(student_middle_0, teacher_middle_0)
            middle_1_loss = middle_L2_loss(student_middle_1, teacher_middle_1)
            middle_2_loss = middle_L2_loss(student_middle_2, teacher_middle_2)
            middle_3_loss = middle_L2_loss(student_middle_3, teacher_middle_3)
        elif block_cfg == 1:
            middle_0_loss = 0
            middle_1_loss = middle_L2_loss(student_middle_1, teacher_middle_1)
            middle_2_loss = middle_L2_loss(student_middle_2, teacher_middle_2)
            middle_3_loss = middle_L2_loss(student_middle_3, teacher_middle_3)
        elif block_cfg == 2:
            middle_0_loss = 0
            middle_1_loss = 0
            middle_2_loss = middle_L2_loss(student_middle_2, teacher_middle_2)
            middle_3_loss = middle_L2_loss(student_middle_3, teacher_middle_3)
        elif block_cfg == 3:
            middle_0_loss = 0
            middle_1_loss = 0
            middle_2_loss = 0
            middle_3_loss = middle_L2_loss(student_middle_3, teacher_middle_3)

            
        loss = alpha * dist_loss + cross_entropy_loss
        if block_cfg == 0:
            loss += middle_0_loss + middle_1_loss + middle_2_loss + middle_3_loss
        elif block_cfg == 1:
            loss += middle_1_loss + middle_2_loss + middle_3_loss
        elif block_cfg == 2:
            loss += middle_2_loss + middle_3_loss
        elif block_cfg == 3:
            loss += middle_3_loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        sample_count += batch_size

        # Compute the number of correctly classified samples
        _, predicted = torch.max(outputs.data, 1)
        # print(labels, predicted)
        train_correct += (predicted == labels).sum().item()
        # print(train_correct)
        epoch_loss += loss.item()
        epoch_distill_loss += dist_loss.item()
        epoch_cross_entropy_loss += cross_entropy_loss.item()

        epoch_acc = train_correct * 100.0 / sample_count
        if block_cfg == 0:
            epoch_middle_0_loss += middle_0_loss.item()
            epoch_middle_1_loss += middle_1_loss.item()
            epoch_middle_2_loss += middle_2_loss.item()
            epoch_middle_3_loss += middle_3_loss.item()
        elif block_cfg == 1:
            epoch_middle_1_loss += middle_1_loss.item()
            epoch_middle_2_loss += middle_2_loss.item()
            epoch_middle_3_loss += middle_3_loss.item()
        elif block_cfg == 2:
            epoch_middle_2_loss += middle_2_loss.item()
            epoch_middle_3_loss += middle_3_loss.item()
        elif block_cfg == 3:
            epoch_middle_3_loss += middle_3_loss.item()

        tmp_loss = epoch_loss * 1.0 / sample_count * batch_size
        tmp_distill_loss = epoch_distill_loss * 1.0 / sample_count * batch_size
        tmp_cross_entropy_loss = epoch_cross_entropy_loss * 1.0 / sample_count * batch_size
        tmp_middle_0_loss = epoch_middle_0_loss * 1.0 / sample_count * batch_size
        tmp_middle_1_loss = epoch_middle_1_loss * 1.0 / sample_count * batch_size
        tmp_middle_2_loss = epoch_middle_2_loss * 1.0 / sample_count * batch_size
        tmp_middle_3_loss = epoch_middle_3_loss * 1.0 / sample_count * batch_size



        # Update the progress bar for the epoch
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({'loss': tmp_loss, 'acc': epoch_acc, 'distill_loss': tmp_distill_loss, 'cross_entropy_loss': tmp_cross_entropy_loss, 'middle_0_loss': tmp_middle_0_loss, 'middle_1_loss': tmp_middle_1_loss, 'middle_2_loss': tmp_middle_2_loss, 'middle_3_loss': tmp_middle_3_loss})
    # Compute the training loss and accuracy for this epoch
    epoch_loss /= (len(train_dataloader.dataset) / batch_size)
    epoch_distill_loss /= (len(train_dataloader.dataset) / batch_size)
    epoch_cross_entropy_loss /= (len(train_dataloader.dataset) / batch_size)
    epoch_middle_0_loss /= (len(train_dataloader.dataset) / batch_size)
    epoch_middle_1_loss /= (len(train_dataloader.dataset) / batch_size)
    epoch_middle_2_loss /= (len(train_dataloader.dataset) / batch_size)
    epoch_middle_3_loss /= (len(train_dataloader.dataset) / batch_size)

    train_accuracy = 100.0 * train_correct / len(train_dataloader.dataset)
    # Close the progress bar for the epoch
    epoch_pbar.close()
    print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.2f}, Training Distill Loss: {epoch_distill_loss:.4f}, Training Cross Entropy Loss: {epoch_cross_entropy_loss:.4f}, Training Middle 0 Loss: {epoch_middle_0_loss:.4f}, Training Middle 1 Loss: {epoch_middle_1_loss:.4f}, Training Middle 2 Loss: {epoch_middle_2_loss:.4f}, Training Middle 3 Loss: {epoch_middle_3_loss:.4f}')
    # Evaluate the model on the validation set
    val_loss = 0.0
    val_correct = 0
    student_model.eval()  # set the model to eval mode

    epoch_pbar = tqdm(desc=f"VAL Epoch {epoch+1}/{num_epochs}",
                    total=len(val_dataloader.dataset) / batch_size, position=0)
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            # Forward pass
            if block_cfg == 0:
                student_middle_0, student_middle_1, student_middle_2, student_middle_3, outputs = student_model(inputs)
            elif block_cfg == 1:
                student_middle_1, student_middle_2, student_middle_3, outputs = student_model(inputs)
            elif block_cfg == 2:
                student_middle_2, student_middle_3, outputs = student_model(inputs)
            elif block_cfg == 3:
                student_middle_3, outputs = student_model(inputs)

            loss = hard_loss(outputs, labels)
            val_loss += loss.item()

            # Compute the number of correctly classified samples
            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            epoch_pbar.update(1)
    epoch_pbar.close()

    # Compute the validation loss and accuracy for this epoch
    val_loss /= (len(val_dataloader.dataset) / batch_size)
    val_accuracy = 100.0 * val_correct / len(val_dataloader.dataset)
    print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    if val_accuracy > val_best_acc:
        val_best_acc = val_accuracy
        torch.save(student_model.state_dict(), out_model_path)
        print('model saved!')


    # update the learning rate
    scheduler.step()
