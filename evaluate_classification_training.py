import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# dt = np.load('loss_experiments/vgg16_classification_losses.npy')
# (train_losses, train_accuracies, val_losses, val_accuracies) = dt
# x_data = np.arange(700)

# plt.plot(x_data, train_losses, label='Training CrossEntropy Loss')
# plt.plot(x_data, val_losses, label='Validation CrossEntropy Loss')
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.savefig('loss_experiments/vgg16_loss_plots')

# plt.close()
# plt.plot(x_data, train_accuracies, label='Training Set Accuracy')
# plt.plot(x_data, val_accuracies, label='Validation Set Accuracy')
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.savefig('loss_experiments/vgg16_accuracy_plots')


dt = np.load('loss_experiments/resnet_classification_losses.npy')
(train_losses, train_accuracies, val_losses, val_accuracies) = dt
x_data = np.arange(len(train_losses))

plt.close()
plt.plot(x_data, train_losses, label='Training CrossEntropy Loss')
plt.plot(x_data, val_losses, label='Validation CrossEntropy Loss')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.savefig('loss_experiments/resnet_loss_plots')


plt.close()
plt.plot(x_data, train_accuracies, label='Training Set Accuracy')
plt.plot(x_data, val_accuracies, label='Validation Set Accuracy')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
plt.savefig('loss_experiments/resnet_accuracy_plots')


# dt = np.load('loss_experiments/baseline_classification_losses.npy')
# (train_losses, train_accuracies, val_losses, val_accuracies) = dt
# x_data = np.arange(len(train_losses))

# plt.close()
# plt.plot(x_data, train_losses, label='Training CrossEntropy Loss')
# plt.plot(x_data, val_losses, label='Validation CrossEntropy Loss')
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.savefig('loss_experiments/baseline_loss_plots')


# plt.close()
# plt.plot(x_data, train_accuracies, label='Training Set Accuracy')
# plt.plot(x_data, val_accuracies, label='Validation Set Accuracy')
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Accuracy')
# plt.savefig('loss_experiments/baseline_accuracy_plots')