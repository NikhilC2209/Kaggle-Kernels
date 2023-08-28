class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(Residual_Block, self).__init__()
        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels))
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Resnet_Model_Torch(nn.Module):
    def __init__(self, block, num_layers_arr, num_classes):
        super(Resnet_Model_Torch, self).__init__()

        #self.expected_input_shape = (1, 3, 224, 224)
        self.residual_channels=64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
    
        self.maxpool = nn.MaxPool2d(kernel_size=2)
    
        #self.test = self.downsample_block(in_channels, out_channels, stride)
        self.layer1 = self.make_layer(block, 64, 64, num_layers_arr[0], stride=1)
        self.layer2 = self.make_layer(block, 64, 128, num_layers_arr[1], stride=2)
        self.layer3 = self.make_layer(block, 128, 256, num_layers_arr[2], stride=2)
        self.layer4 = self.make_layer(block, 256, 512, num_layers_arr[3], stride=2)
    
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.Flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)
    
    def downsample_block(self, in_channels, out_channels, stride):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),                    # 1x1 convolution with stride=2 to halve size
            nn.BatchNorm2d(out_channels)
        )
    
    def make_layer(self, block, in_channels, out_channels, num_layers, stride): 
        downsample = None
        if stride != 1:
            downsample = self.downsample_block(in_channels, out_channels, stride)
            
        layers = []
        layers.append(block(self.residual_channels, out_channels, stride, downsample))
        self.residual_channels=out_channels
        for i in range(1, num_layers):
            layers.append(block(self.residual_channels, out_channels))
            
        return nn.Sequential(*layers)    

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.Flatten(out)
        out = self.fc(out)

        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Resnet_Model_Torch(Residual_Block, [2, 2, 2, 2], 75).to(device)                  ### ResNet18 Architecture
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)


### PRINT MODEL LAYERS

print(model)

from torchsummary import summary
#summary(model, (3, 224, 224))
summary(model, input_image_tensor_shape)