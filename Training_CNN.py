### TRAINING SCRIPT 

num_epochs = 20
train_loss_list = []
val_loss_list = []
train_acc = 0
count1 = len(train_dataloader.dataset)
count2 = len(val_dataloader.dataset)

for epoch in range(num_epochs):
  epoch_acc = 0
  train_loss = 0
  val_loss = 0
  val_epoch_acc = 0
  for images, labels in train_dataloader:
    #images = images.view(-1,1,28,28)
    #count += len(labels)

    images, labels = images.to(device), labels.to(device)

    images = images.float()
    y_pred = model(images)
    loss = loss_fn(y_pred, labels)

    optim.zero_grad()
    loss.backward()
    optim.step()

    train_loss += loss.item()
    epoch_acc += (y_pred.argmax(1) == labels).sum().item()

  ### Validation script

  with torch.no_grad():
    model.eval()

    for val_images, val_labels in val_dataloader:
        
        val_images, val_labels = val_images.to(device), val_labels.to(device)
        
        val_images = val_images.float()
        y_pred_val = model(val_images)
        loss = loss_fn(y_pred_val, val_labels)

        val_loss += loss.item()
        val_epoch_acc += (y_pred_val.argmax(1) == val_labels).sum().item()

  print('Epoch: {} - Loss: {:.6f}, Training Acc.: {:.3%}, Val. Loss: {:.6f}, Validation Acc.: {:.3%}'.format(epoch + 1, train_loss/trainSteps, epoch_acc/count1, val_loss/valSteps, val_epoch_acc/count2))
  train_loss_list.append(train_loss/trainSteps)
  val_loss_list.append(val_loss/valSteps)
  #print('Epoch: {} - Accuracy: {:.3%}'.format(epoch + 1, epoch_acc/count))
  #train_acc += epoch_acc
  #train_loss += loss
  #train_acc += (y_pred.argmax(1) == labels).sum()


### PERFORMANCE PLOTS

plt.plot(train_loss_list)
plt.plot(val_loss_list)
# plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()
