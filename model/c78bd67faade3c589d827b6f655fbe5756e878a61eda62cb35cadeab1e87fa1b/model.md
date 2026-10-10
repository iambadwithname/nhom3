SimpleCNN(
  (CNN): Sequential(
    (0): Conv2d(1, 2, kernel_size=(3, 3), stride=(1, 1))
    (1): Conv2d(2, 4, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3))
    (2): MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1, ceil_mode=False)
    (3): Conv2d(4, 8, kernel_size=(3, 3), stride=(3, 3))
    (4): MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1, ceil_mode=False)
    (5): Conv2d(8, 16, kernel_size=(3, 3), stride=(3, 3))
    (6): MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1, ceil_mode=False)
    (7): Conv2d(16, 32, kernel_size=(3, 3), stride=(3, 3))
    (8): Conv2d(32, 64, kernel_size=(3, 3), stride=(3, 3))
  )
  (pooling): AdaptiveAvgPool2d(output_size=(2, 2))
  (Activation): Sequential(
    (0): ReLU()
  )
  (fc): Sequential(
    (0): Linear(in_features=256, out_features=64, bias=True)
    (1): Sequential(
      (0): ReLU()
    )
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): Sequential(
      (0): ReLU()
    )
    (4): Linear(in_features=32, out_features=3, bias=True)
  )
)