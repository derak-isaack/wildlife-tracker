import torch 

device = torch.device('cpu')
# device.load_state_dict(torch.load('wildlife5_vision.pth', map_location=device))
state_dict = torch.load('wildlife5_vision.pth', map_location=device)
print(state_dict.keys())



