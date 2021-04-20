import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# dt = np.load('loss_experiments/vgg16_losses_adaptive.npy')
# (train_losses, train_r2, val_losses, val_r2) = dt
# x_data = np.arange(700)

# plt.plot(x_data, train_losses, label='Training MSE Loss')
# plt.plot(x_data, val_losses, label='Validation MSE Loss')
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Loss')
# plt.savefig('loss_experiments/vgg16_regression_loss_plots')

# plt.close()
# plt.plot(x_data, train_r2, label='Training Set r2')
# plt.plot(x_data, val_r2, label='Validation Set r2')
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('r2')
# plt.savefig('loss_experiments/vgg16_regression_r2_plots')


dt = np.load('loss_experiments/resnet18denmap/resnet18_losses.npy')
(train_losses, train_r2, val_losses, val_r2) = dt
x_data = np.arange(len(train_losses))

plt.close()
plt.plot(x_data, train_losses, label='Training MSE Loss')
plt.plot(x_data, val_losses, label='Validation MSE Loss')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.savefig('loss_experiments/resnet18denmap/resnet18_regression_loss_plots')

print(val_r2)
print(train_r2)

plt.close()
plt.plot(x_data, train_r2, label='Training Set r2')
plt.plot(x_data, val_r2, label='Validation Set r2')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('r2')
plt.savefig('loss_experiments/resnet18denmap/resnet18_regression_r2_plots')