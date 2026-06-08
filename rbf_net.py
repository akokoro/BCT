import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PINN_module(nn.Module):
    def __init__(self, n_in, n_out, n_neu_x, n_neu_y, b, c_x, c_y, device):
        super(PINN_module, self).__init__()
        self.regularizer = None
        self.n_in = n_in
        self.n_out = n_out
        self.n_neu_x = n_neu_x
        self.n_neu_y = n_neu_y
        self.k_neighbors = int(1024)
        self.b = b
        self.device = device
        self.c, self.inputs_len = self.reviseC(c_x, c_y)

        self.net = LocalRBF(
            n_in=n_in, 
            n_out=n_out, 
            n_neu=self.inputs_len,
            centers=self.c,
            k_neighbors=self.k_neighbors,
            base_b=self.b
        )

    def reviseC(self, c_x, c_y):
        c = np.zeros((2, self.n_neu_x * self.n_neu_y)).astype(dtype='float32')
        k = 0
        dx = (c_x[1] - c_x[0]) / (self.n_neu_x - 1)
        dy = (c_y[1] - c_y[0]) / (self.n_neu_y - 1)
        for i in range(self.n_neu_x):
            for j in range(self.n_neu_y):
                c[0, k] = i * dx + c_x[0]
                c[1, k] = j * dy + c_y[0]
                k = k + 1
        # distances = np.sqrt(c[0] ** 2 + c[1] ** 2)
        # indices_inside_circle = np.where(distances < R)[0]
        # c = np.delete(c, indices_inside_circle, axis=1)
        # plt.figure(figsize=(8, 6))
        # plt.scatter(c[0, :], c[1, :], c='blue', marker='o', s=1)
        # plt.colorbar(label='B')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('111')
        # plt.show()
        length =c.shape[1]
        c = torch.tensor(c).to(self.device)
        return c, length

    def init_weights(self, layers):
        b = (np.ones((1, self.inputs_len)) * self.b).astype(np.float32)
        b = torch.from_numpy(b).to(self.device).requires_grad_(True)
        layers[0].b.data = b

    def forward(self, x):
         y = self.net(x)
         return y

class LocalRBF(nn.Module):
    def __init__(self, n_in, n_out, n_neu, centers, k_neighbors, base_b):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_neu = n_neu
        self.k_neighbors = k_neighbors
        self.b = nn.Parameter(torch.ones(1, self.n_neu) * base_b)

        self.register_buffer('c', centers)
        self.output_weights = nn.Parameter(torch.randn(self.n_neu, self.n_out))
        nn.init.xavier_uniform_(self.output_weights)
        
        self.forward_count = 0
        self.time_distance = 0.0
        self.time_topk = 0.0
        self.time_gather_activation = 0.0
        self.time_gather_weights = 0.0
        self.time_total = 0.0
        
        self.backward_count = 0
        self.time_backward = 0.0
        self.backward_start_event = None
        self.backward_end_event = None
        
        self.register_full_backward_hook(self._backward_hook)

    def forward(self, inputs):
        if inputs.is_cuda:
            start_total = torch.cuda.Event(enable_timing=True)
            end_total = torch.cuda.Event(enable_timing=True)
            start_1 = torch.cuda.Event(enable_timing=True)
            end_1 = torch.cuda.Event(enable_timing=True)
            start_2 = torch.cuda.Event(enable_timing=True)
            end_2 = torch.cuda.Event(enable_timing=True)
            start_3 = torch.cuda.Event(enable_timing=True)
            end_3 = torch.cuda.Event(enable_timing=True)
            start_4 = torch.cuda.Event(enable_timing=True)
            end_4 = torch.cuda.Event(enable_timing=True)
            
            start_total.record()
            
            start_1.record()
        
        t2 = (inputs[..., 0, None] ** 2 + inputs[..., 1, None] ** 2)
        D = (self.c[None, 0, :] ** 2 + self.c[None, 1, :] ** 2)
        t1 = (2 * torch.matmul(inputs, self.c))
        distances_squared = (t2 + D - t1)
        
        if inputs.is_cuda:
            end_1.record()
            
            start_2.record()

        _, indices = torch.topk(-distances_squared, k=self.k_neighbors, dim=1)  # [batch, k]

        if inputs.is_cuda:
            end_2.record()
            
            start_3.record()

        selected_distances = torch.gather(distances_squared, 1, indices)  # [batch, k]
        selected_b = torch.gather(self.b.expand(inputs.shape[0], -1), 1, indices)  # [batch, k]
        selected_activations = torch.exp(-selected_distances * selected_b**2)  # [batch, k]

        if inputs.is_cuda:
            end_3.record()
            
            start_4.record()

        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_out)  # [batch, k, n_out]
        batch_weights = self.output_weights.unsqueeze(0).expand(inputs.shape[0], -1, -1)  # [batch, n_neu, n_out]
        selected_weights = torch.gather(batch_weights, 1, expanded_indices)  # [batch, k, n_out]
        
        output = selected_activations.unsqueeze(-1) * selected_weights  # [batch, k, n_out]
        final_output = torch.sum(output, dim=1)
        
        if inputs.is_cuda:
            end_4.record()
            end_total.record()
            
            torch.cuda.synchronize()
            self.time_distance += start_1.elapsed_time(end_1)
            self.time_topk += start_2.elapsed_time(end_2)
            self.time_gather_activation += start_3.elapsed_time(end_3)
            self.time_gather_weights += start_4.elapsed_time(end_4)
            self.time_total += start_total.elapsed_time(end_total)
            self.forward_count += 1
            
            self.backward_start_event = torch.cuda.Event(enable_timing=True)
            self.backward_end_event = torch.cuda.Event(enable_timing=True)
            self.backward_start_event.record()
        
        return final_output
    
    def _backward_hook(self, module, grad_input, grad_output):
        """hookbackward"""
        if self.backward_end_event is not None:
            self.backward_end_event.record()
            torch.cuda.synchronize()
            self.time_backward += self.backward_start_event.elapsed_time(self.backward_end_event)
            self.backward_count += 1
    
    def get_timing_stats(self):
        """ (: ms)"""
        if self.forward_count == 0:
            return None
        return {
            'forward_count': self.forward_count,
            'backward_count': self.backward_count,
            'avg_distance_ms': self.time_distance / self.forward_count,
            'avg_topk_ms': self.time_topk / self.forward_count,
            'avg_gather_activation_ms': self.time_gather_activation / self.forward_count,
            'avg_gather_weights_ms': self.time_gather_weights / self.forward_count,
            'avg_total_forward_ms': self.time_total / self.forward_count,
            'avg_backward_ms': self.time_backward / self.backward_count if self.backward_count > 0 else 0.0,
            'total_distance_ms': self.time_distance,
            'total_topk_ms': self.time_topk,
            'total_gather_activation_ms': self.time_gather_activation,
            'total_gather_weights_ms': self.time_gather_weights,
            'total_forward_ms': self.time_total,
            'total_backward_ms': self.time_backward,
        }
    
    def reset_timing_stats(self):
        """"""
        self.forward_count = 0
        self.backward_count = 0
        self.time_distance = 0.0
        self.time_topk = 0.0
        self.time_gather_activation = 0.0
        self.time_gather_weights = 0.0
        self.time_total = 0.0
        self.time_backward = 0.0
