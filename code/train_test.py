import numpy as np
import os
import datetime
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  
from sklearn.metrics import f1_score, accuracy_score

dir = os.path.join('runs', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

def training(modeltype, batch_size, n_epoch, criterion, optimizer, scheduler, train_loader, valid_loader, model, device, early_stop_num):
    total = sum(p.numel() for p in model.parameters())  # All parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) # Trainable parameters
    print('start training, parameter total:{}, trainable:{}'.format(total, trainable))

    model.train() # Let optimizer update the parameters
    t_batch = len(train_loader) 
    v_batch = len(valid_loader) 
    log_writer = SummaryWriter(dir) # Write log file
    best_acc, batch_num, continue_bigger_num = 0, 0, 0

    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0

        for i, (inputs, labels) in enumerate(train_loader):
            batch_num += 1
            inputs = inputs.to(device, dtype=torch.long) # transform inputs to torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # transform inputs to torch.cuda.FloatTensor
            optimizer.zero_grad() # Reset gradient to zero
            if modeltype == 'BERT':
                output = model(inputs, token_type_ids=None, attention_mask=inputs>0, labels=labels)
                outputs = output.logits.squeeze()
                loss = output.loss
                # logits = output.logits.detach().cpu().numpy()
                # labels = labels.detach().cpu().numpy()
                # tmp_acc = flat_accuracy(logits, labels)

            else:
                outputs = model(inputs)
                outputs = outputs.squeeze() # deleter the outmost dimension
                loss = criterion(outputs, labels)
            loss.backward() # Get the loss gradient
            optimizer.step() # Update model training parameters
            if modeltype == 'BERT':
                scheduler.step()
            tmp_acc = evaluation(outputs, labels) / batch_size
            total_acc += tmp_acc
            total_loss += loss.item()
            log_writer.add_scalar('Loss/Train', float(loss), batch_num) # Draw in Tensorboard
            log_writer.add_scalar('Acc/Train', float(tmp_acc*100), batch_num) # Draw in Tensorboard
            # print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(epoch+1, i+1, t_batch, loss.item(), tmp_acc*100), end='\r')
        # print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))
        print('\nEpoch{}: {}/{} \nTrain | Loss:{:.5f} Acc: {:.3f}'.format(epoch+1, i+1, t_batch, total_loss/t_batch, total_acc/t_batch*100))

        # Validation
        model.eval() # Fix the model parameters
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid_loader):
                inputs = inputs.to(device, dtype=torch.long) 
                labels = labels.to(device, dtype=torch.float)  
                if modeltype == 'BERT':
                    output = model(inputs, token_type_ids=None, attention_mask=(inputs>0), labels=labels)
                    outputs = output.logits.squeeze()
                    loss = output.loss
                    # logits = output.logits.detach().cpu().numpy()
                    # labels = labels.detach().cpu().numpy()
                    # tmp_acc = flat_accuracy(logits, labels)
                else:
                    outputs = model(inputs) 
                    outputs = outputs.squeeze() 
                    loss = criterion(outputs, labels) 
                tmp_acc = evaluation(outputs, labels) / batch_size
                total_acc += tmp_acc
                total_loss += loss.item()

            log_writer.add_scalar('Loss/Validation', float(total_loss/v_batch), epoch) # Write in Tensorboard
            log_writer.add_scalar('Acc/Validation', float(total_acc/v_batch*100), epoch) # Write in Tensorboard
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                best_acc = total_acc
                continue_bigger_num = 0
                total_acc_new = total_acc/v_batch*100
                if modeltype == 'LSTM' and total_acc_new > 78 or\
                    modeltype == 'CNN' and total_acc_new > 79 or\
                    modeltype == 'BERT' and total_acc_new > 81:
                    torch.save(model, "./models/" + modeltype + "/ckpt_" + str(round(total_acc_new, 3)) + ".model")
                    print('saving model with acc {:.3f}'.format(total_acc_new))
            else:
                continue_bigger_num += 1
                if continue_bigger_num == early_stop_num:
                    print("EARLY STOP SATISFIES, STOP TRAINING")
                    break
        print('-----------------------------------------------')
        model.train() # Let optimizer update the parameters


def testing(modeltype, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            if modeltype == 'BERT':
                output = model(inputs, token_type_ids=None, attention_mask=(inputs>0))
                outputs = output.logits.squeeze()
            else:
                outputs = model(inputs)
                outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # More than 0.5 set to real
            outputs[outputs<0.5] = 0 # Less than 0.5 set to fake
            ret_output += outputs.int().tolist()
    
    return ret_output

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    pred = torch.zeros_like(outputs)
    pred[outputs >= 0.5] = 1
    pred[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(pred, labels)).item()
    return correct

# def flat_accuracy(preds, labels):
    
#     """A function for calculating accuracy scores"""
    
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return accuracy_score(labels_flat, pred_flat)
#     # print(pred_flat.shape, labels_flat.shape)
#     # return np.sum(pred_flat == labels_flat)