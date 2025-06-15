## Vision Transformer (ViT) 实现
这个项目展示了如何使用PyTorch实现一个基本的Vision Transformer（ViT）模型，并在自定义数据集上进行训练和验证。

## 项目结构
day3/dataset.py: 自定义数据集类 ImageTxtDataset，用于读取图片及标签。
main.py: ViT模型的实现及训练和评估流程。
logs_vit_rewrite: TensorBoard日志存储目录。
model_save: 保存训练过程中模型的目录。

## 依赖
Python 3.x
PyTorch >= 1.8
torchvision
einops
TensorBoard

## 安装依赖：
bash
pip install torch torchvision einops tensorboard

## 数据集
该模型使用自定义的图片和标签文件集（.txt）进行训练。数据集路径如下：
train.txt: 包含训练集图像路径和标签。
val.txt: 包含验证集图像路径和标签。
图像存储在 train_img_dir 和 val_img_dir 目录下。

## 配置
在 Config 类中配置以下参数：
```python
class Config:
    train_txt = r"D:\intership\day3\data\train.txt"
    train_img_dir = r"D:\intership\day3\data\Images\train"
    val_txt   = r"D:\intership\day3\data\val.txt"
    val_img_dir   = r"D:\intership\day3\data\Images\val"
    batch_size = 64
    lr = 1e-4
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "logs_vit_rewrite"
    model_dir = "model_save"
```
## 模型架构
实现了一个标准的Vision Transformer（ViT）架构，主要包括以下部分：
Patch Embedding: 将图像分成固定大小的块，并进行嵌入。
Transformer: 使用自注意力机制（Attention）和前馈网络（FeedForward）层堆叠形成Transformer。
分类头: 通过线性层进行分类。

## 主要模块
1. FeedForward 类
一个简单的前馈神经网络，用于Transformer的MLP部分。
2. Attention 类
实现了多头自注意力机制，用于捕捉图像的全局上下文信息。
3. Transformer 类
由多个 Attention 和 FeedForward 层堆叠而成。
4. ViT 类
结合了图像的嵌入、位置编码和Transformer层进行分类。

## 训练与评估
## 训练
```python
def train_one_epoch(epoch, step):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(Config.device), labels.to(Config.device)
        imgs = imgs.squeeze(2)  # (B, C, 1, 256) → (B, C, 256)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
## 验证
```python
def evaluate(epoch):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(Config.device), labels.to(Config.device)
            imgs = imgs.squeeze(2)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    acc = correct / len(val_dataset)
    print(f"[Eval] Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")
```
## 主训练循环
for epoch in range(Config.epochs):
    print(f"\n===== 第 {epoch+1} 轮训练开始 =====")
    total_step = train_one_epoch(epoch, total_step)
    evaluate(epoch)
    os.makedirs(Config.model_dir, exist_ok=True)
    model_path = os.path.join(Config.model_dir, f"vit_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")

## 日志与模型保存
训练过程中，使用TensorBoard记录训练损失、验证损失和准确率，可以通过以下命令查看：
bash
tensorboard --logdir=logs_vit_rewrite
训练完成后，模型会保存到 model_save 目录下。

## 结果
训练结束后，模型的最佳性能会存储为 vit_epochX.pth 文件。