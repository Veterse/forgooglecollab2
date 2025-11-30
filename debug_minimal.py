
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Minimal Network for reproduction
class MinimalNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 18 channels, 8x8
        self.conv1 = nn.Conv2d(18, 32, 3)
        self.fc_policy = nn.Linear(32 * 6 * 6, 4672)
        self.fc_value = nn.Linear(32 * 6 * 6, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        
        policy = self.fc_policy(x)
        value = self.fc_value(x)
        
        return F.log_softmax(policy, dim=1), torch.tanh(value)

def run_minimal_test():
    print("--- Minimal CPU Test ---")
    device = "cpu"
    model = MinimalNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy Data
    batch_size = 10
    state = torch.randn(batch_size, 18, 8, 8).to(device)
    
    # Target: Always index 100 is the move
    policy_target = torch.zeros(batch_size, 4672).to(device)
    policy_target[:, 100] = 1.0
    
    value_target = torch.ones(batch_size, 1).to(device)
    
    # Initial prediction
    model.eval()
    with torch.no_grad():
        log_p, v = model(state)
        prob = torch.exp(log_p[0, 100]).item()
        print(f"Start: Prob[100]={prob:.6f}, Val={v[0].item():.6f}")
        
    # Train loop
    model.train()
    for i in range(50):
        optimizer.zero_grad()
        log_p, v = model(state)
        
        # Value Loss (MSE)
        v_loss = F.mse_loss(v, value_target)
        
        # Policy Loss (Cross Entropy with soft targets)
        # -sum(target * log_pred)
        p_loss = -torch.sum(policy_target * log_p) / batch_size
        
        loss = v_loss + p_loss
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Iter {i}: Loss={loss.item():.4f}")
            
    # Final prediction
    model.eval()
    with torch.no_grad():
        log_p, v = model(state)
        prob = torch.exp(log_p[0, 100]).item()
        print(f"End: Prob[100]={prob:.6f}, Val={v[0].item():.6f}")

    if prob > 0.9:
        print("SUCCESS: Minimal logic works.")
    else:
        print("FAILURE: Minimal logic broken.")

if __name__ == "__main__":
    run_minimal_test()
