import matplotlib.pyplot as plt

def plot_metrics(losses, accuracies):
    '''
    绘制模型训练过程中的损失函数和准确率图。

    Args:
        losses (list): 模型在每个 Epoch 结束时的训练损失值列表。
        accuracies (list): 模型在每个 Epoch 结束时的训练准确率值列表。

    Returns:
        None: 函数直接调用 plt.show() 显示图表。
    '''
    epochs = range(1, len(losses) + 1)
    
    # --- 绘制损失函数图 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1) # 1行2列的图，这是第1个
    plt.plot(epochs, losses, 'b', label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # --- 绘制准确率图 ---
    plt.subplot(1, 2, 2) # 1行2列的图，这是第2个
    plt.plot(epochs, accuracies, 'r', label='Training Accuracy')
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
    plt.show() # 显示图表

def visualize_images(images, labels, predictions):
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))  
    axes = axes.ravel()  

    for i in range(6):
        image = images[i].squeeze()  
        ax = axes[i]
        ax.imshow(image, cmap='gray')  
        ax.set_title(f"Pred: {predictions[i]} | Actual: {labels[i]}")  
        ax.axis('off')  

    plt.tight_layout() 
    plt.show()  