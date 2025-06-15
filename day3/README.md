这个项目展示了如何使用PyTorch和TensorBoard可视化非线性激活函数（特别是Sigmoid）对图像的影响。具体而言，它使用CIFAR-10数据集，并记录了在应用Sigmoid激活函数前后图像的变化。

## 项目核心：
Sigmoid 激活函数：它将输入值压缩到[0, 1]的范围。通过将CIFAR-10图像应用Sigmoid函数，输出图像的亮度会变得更暗，因为Sigmoid会压缩像素值。
TensorBoard 记录和可视化：原始图像和经过Sigmoid激活后的图像都会被记录在TensorBoard中，用户可以通过TensorBoard来对比这两者的差异。

## 步骤：
加载CIFAR-10测试集（默认没有标签）。
对图像应用Sigmoid激活函数。
保存处理前后的图像到TensorBoard日志中。

## 结果：
通过TensorBoard，你可以看到应用Sigmoid前后的图像效果：
输入：原始的CIFAR-10图像。
输出：应用Sigmoid后的图像，它们通常会显得更暗，因为Sigmoid压缩了图像的像素值。

## 核心代码：
```python
class Chen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        return self.sigmoid(input)
 ```              
这个简单的网络只包含Sigmoid激活函数，用户可以将其替换为其他激活函数（如ReLU或Tanh）来观察不同的效果。