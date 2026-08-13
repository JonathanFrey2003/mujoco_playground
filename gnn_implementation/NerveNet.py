import dataclasses
import functools
from typing import Any, Callable, Literal, Mapping, Sequence, Tuple
import warnings
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.spectral_norm import SNDense
from flax import linen
from flax import linen as nn
import jax
import jax.numpy as jnp

class GNN(nn.Module):
    layer_sizes: Sequence[int] 
    hidden_dim: int
    message_passing_steps: int
    layer_norm: bool = False
    activation: ActivationFn = nn.relu
    edges: jnp.ndarray = dataclasses.field(
        default_factory=lambda: jnp.array(
            [[0, 1], [1, 2], [2, 3],
             [3, 4], [0, 5], [5, 6],
             [6, 7], [7, 8], [0, 9],
             [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15],
             [15, 16]],
            dtype=jnp.int32,
        )
    )
    adjacency_matrix: jnp.ndarray = None  # shape (n_nodes, n_nodes)
    num_nodes: int = None 
    node_feature_dim: int = None
    

    def setup(self):
      default_adjacency_matrix = jnp.zeros((self.num_nodes, self.num_nodes))

      for i in range(self.edges.shape[0]):
        default_adjacency_matrix = default_adjacency_matrix.at[self.edges[i, 0], self.edges[i, 1]].set(1)
        default_adjacency_matrix = default_adjacency_matrix.at[self.edges[i, 1], self.edges[i, 0]].set(1)
         
      self._adj_list = default_adjacency_matrix if self.adjacency_matrix is None else self.adjacency_matrix

      self.input_model = [MLP([128, self.hidden_dim], layer_norm=False, activation=self.activation)
                          for _ in range(self.num_nodes)]
      
      self.message_network = MLP(layer_sizes=[128,self.hidden_dim],
                                layer_norm=False, 
                                activation=self.activation,
                                activate_final=True
                                 )
      
      self.update_network = MLP(layer_sizes=[128, self.hidden_dim], 
                               layer_norm=False,
                               activation=self.activation,
                               activate_final=True
                               )

      self.action_decoders = [MLP([128, 2], layer_norm=False, activation=self.activation)
                              for _ in range(self.num_nodes-1)]
      
    
    @linen.compact
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:     
      # Turn node features into graph structure
      batch_shape = data.shape[:-1]
      #print("input data shape:", data.shape) # should be (batch_size, num_obs_sensor * node_feature_dim)??
      node_states_list = []
      for i in range(self.num_nodes):
        node_states_list.append(self.input_model[i](data))
      
      node_states = jnp.stack(node_states_list, axis=-2) # 

      # Message passing (Propagation model)
      for _ in range(self.message_passing_steps):
        
        # Message Computation
        messages = self.message_network(node_states) # Eq.2

        # Message Aggregation (sum)
        aggregated_messages = jnp.matmul(self._adj_list, messages) 
          
        # Update node states
        update_inputs = jnp.concatenate([node_states, aggregated_messages], axis=-1)
        node_states = self.update_network(update_inputs) # Eq.3 (batch_size, num_nodes, hidden_dim)

      # Output model
      actions = []
      for node_idx in range(self.num_nodes-1):
        joint_idx = node_idx + 1 # since node_idx starts from 0, and the root node is at index 0
        a_i = self.action_decoders[node_idx](node_states[..., joint_idx, :]) # Eq.4 (batch_size, output_dim)
        actions.append(a_i)

      x = jnp.concatenate(
          actions,
          axis=-1)

      #print("output x shape:", x.shape)
      return x


class GNN_old(nn.Module):
    layer_sizes: Sequence[int] 
    hidden_dim: int
    message_passing_steps: int
    layer_norm: bool = False
    activation: ActivationFn = nn.relu
    edges: jnp.ndarray = None  # shape (n_edges, 2)
    num_nodes: int = None 
    node_feature_dim: int = None
    

    def setup(self):
      default_edges = jnp.array([[0,1],[1,2],[2,3],
                              [3,4],[0,5],[5,6],
                              [6,7],[7,8],[0,9],
                              [9,10],[10,11],[11,12],
                              [0,13],[13,14],[14,15],
                              [15,16]])
      
      self._edges = self.edges if self.edges is not None else default_edges

      self.input_model = [MLP([self.hidden_dim], layer_norm=False, activation=self.activation)
                          for _ in range(self.num_nodes)]
      
      self.message_network = MLP(layer_sizes=[128,self.hidden_dim],
                                layer_norm=False, 
                                activation=self.activation,
                                activate_final=True
                                 )
      
      self.update_network = MLP(layer_sizes=[128, self.hidden_dim], 
                               layer_norm=False,
                               activation=self.activation,
                               activate_final=True
                               )

      self.action_decoders = [MLP([64, 2], layer_norm=False, activation=self.activation)
                              for _ in range(self.num_nodes-1)]
      
      self.senders = jnp.concatenate([self._edges[:, 0], self._edges[:,1]], axis=0)
      self.receivers = jnp.concatenate([self._edges[:, 1], self._edges[:,0]], axis=0)
    
    @linen.compact
    def __call__(self, data: jnp.ndarray) -> jnp.ndarray:     
      # Turn node features into graph structure
      batch_shape = data.shape[:-1]
      #print("input data shape:", data.shape) # should be (batch_size, num_obs_sensor * node_feature_dim)??
      #node_features = jnp.reshape(data,batch_shape + (self.num_nodes-1, self.node_feature_dim)) 

      #Zero padding for root node
      #node_features = jnp.concatenate([jnp.zeros(batch_shape + (1, self.node_feature_dim)), node_features], axis=-2)

      # Initial node embedding (state vector)
      #print("node feature example", node_features[0])
      #node_states = self.input_model(node_features) # Eq.1, (batch_size, num_nodes, hidden_dim)
      #print("node state vector shape:", node_states.shape

      #feed all data through each encoder for each node
      node_states_list = []
      for i in range(self.num_nodes):
        node_states_list.append(self.input_model[i](data))
      
      node_states = jnp.stack(node_states_list, axis=-2) # Eq.1, (batch_size, num_nodes, hidden_dim)
      node_states = jnp.reshape(node_states,batch_shape + (self.num_nodes, self.hidden_dim)) 
      
      # Message passing (Propagation model)
      for _ in range(self.message_passing_steps):
        
        sender_states = jnp.take(node_states, self.senders, axis = -2)
        receiver_states = jnp.take(node_states, self.receivers, axis = -2)

        # Message Computation
        combined_edge_inputs = jnp.concatenate([sender_states, receiver_states], axis=-1) # (batch_size, num_edges, 2*hidden_dim)
        messages = self.message_network(combined_edge_inputs) # Eq.2 (batch_size, num_edges*2 , hidden_dim)
        #print("messages shape:", messages.shape) 

        # Message Aggregation (sum)
        aggregated_messages = jnp.zeros_like(node_states)
        def sum_messages(msg):
            return jnp.zeros(node_states.shape).at[..., self.receivers, :].add(msg)
        
        aggregated_messages = sum_messages(messages) 
        #print("aggregated messages shape:", aggregated_messages.shape)
          
        # Update node states
        update_inputs = jnp.concatenate([node_states, aggregated_messages], axis=-1)
        node_states = self.update_network(update_inputs) # Eq.3 (batch_size, num_nodes, hidden_dim)
        # node_states, _ = jax.vmap(
        #                 lambda carry, inp: self.update_network(carry, inp),
        #                 in_axes=1,
        #                 out_axes=1,
        #                 )(node_states, aggregated_messages)
        
      # Output model
      actions = []
      for node_idx in range(self.num_nodes-1):
        joint_idx = node_idx + 1 # since node_idx starts from 0, and the root node is at index 0
        a_i = self.action_decoders[node_idx](node_states[..., joint_idx, :]) # Eq.4 (batch_size, output_dim)
        actions.append(a_i)

      x = jnp.concatenate(
          actions,
          axis=-1)

      #print("output x shape:", x.shape)
      return x