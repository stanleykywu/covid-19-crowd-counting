import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

dt = np.load('loss_experiments/vgg16_classification_losses.npy')
(train_losses, train_accuracies, val_losses, val_accuracies) = dt
x_data = np.arange(700)

plt.plot(x_data, train_losses, label='Training CrossEntropy Loss')
plt.plot(x_data, val_losses, label='Validation CrossEntropy Loss')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.savefig('loss_experiments/vgg16_loss_plots')

plt.close()
plt.plot(x_data, train_accuracies, label='Training Set Accuracy')
plt.plot(x_data, val_accuracies, label='Validation Set Accuracy')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.savefig('loss_experiments/vgg16_accuracy_plots')


dt = np.load('loss_experiments/resnet50_classification_losses.npy')
(train_losses, train_accuracies, val_losses, val_accuracies) = dt
x_data = np.arange(700)

plt.close()
plt.plot(x_data, train_losses, label='Training CrossEntropy Loss')
plt.plot(x_data, val_losses, label='Validation CrossEntropy Loss')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.savefig('loss_experiments/resnet50_loss_plots')


plt.close()
plt.plot(x_data, train_accuracies, label='Training Set Accuracy')
plt.plot(x_data, val_accuracies, label='Validation Set Accuracy')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.savefig('loss_experiments/resnet50_accuracy_plots')