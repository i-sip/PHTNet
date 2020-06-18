# PHTNet
**Article:** *PHTNet: Characterization and Deep Mining of Involuntary Pathological Hand Tremor using Recurrent Neural Network Models*

**Bibliography:** [*Soroosh Shahtalebi, Seyed Farokh Atashzar, Olivia Samotus, Rajni V. Patel, Mandar S. Jog, and Arash Mohammadi. "PHTNet: characterization and Deep Mining of involuntary pathological Hand tremor using Recurrent neural network Models." _Scientific reports_ 10, no. 1 (2020): 1-19.*](https://www.nature.com/articles/s41598-020-58912-9.pdf)

**Abstract:** The global aging phenomenon has increased the number of individuals with age-related neurological movement disorders including Parkinson’s Disease (PD) and Essential Tremor (ET). Pathological Hand Tremor (PHT), which is considered among the most common motor symptoms of such disorders, can severely affect patients’ independence and quality of life. To develop advanced rehabilitation and assistive technologies, accurate estimation/prediction of nonstationary PHT is critical, however, the required level of accuracy has not yet been achieved. The lack of sizable datasets and generalizable modeling techniques that can fully represent the spectrotemporal characteristics of PHT have been a critical bottleneck in attaining this goal. This paper addresses this unmet need through establishing a deep recurrent model to predict and eliminate the PHT component of hand motion. More specifically, we propose a machine learning-based, assumption-free, and real-time PHT elimination framework, the PHTNet, by incorporating deep bidirectional recurrent neural networks. The PHTNet is developed over a hand motion dataset of 81 ET and PD patients collected systematically in a movement disorders clinic over 3 years. The PHTNet is the first intelligent systems model developed on this scale for PHT elimination that maximizes the resolution of estimation and allows for prediction of future and upcoming sub-movements.

## Implementation Details

1. Please refere to the file `spec-file.txt` to creat a Python environment. On Anaconda, run the following script:
`conda create --name myenv --file spec-file.txt`
2. Construct a tensorflow `model` according to:
```ruby
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = len(train_final)
batch_size = 1
sampling_freq = 100
batch_length = 4
tf.reset_default_graph()
num_time_steps = 400
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])
n_layers = 4
neurons=[400,400,400,400]
cells = []
for _ in range(n_layers):
  cell = tf.contrib.rnn.GRUCell(400)  # Or LSTMCell(num_units)
  cells.append(cell)
cell = tf.contrib.rnn.MultiRNNCell(cells)
outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw= cell,
                                                  inputs= X, dtype=tf.float32) 
with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
with tf.name_scope("Train"):
    train = optimizer.minimize(loss)

```
3. Load the parameters of the trained model from the file `model.ckpt`
