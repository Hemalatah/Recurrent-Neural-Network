# Recurrent-Neural-Network
Recurrent Neural Network in numpy

Recurrent Neural Networks (RNN) are very effective for Natural Language Processing and other sequence tasks because they have "memory". They can read inputs  x⟨t⟩  (such as words) one at a time, and remember some information/context through the hidden layer activations that get passed from one time-step to the next. This allows a uni-directional RNN to take information from the past to process later inputs. A bidirection RNN can take context from both the past and the future.


Notation:

Superscript  [l]  denotes an object associated with the  lth  layer.

Example:  a[4]  is the  4th4th  layer activation.  W[5]  and  b[5]  are the  5th  layer parameters.
Superscript  (i)  denotes an object associated with the  ith  example.

Example:  x(i)  is the  ith  training example input.
Superscript  ⟨t⟩  denotes an object at the  tth  time-step.

Example:  x⟨t⟩  is the input x at the  tth  time-step.  x(i)⟨t⟩  is the input at the  tth  timestep of example  i .
Lowerscript  i  denotes the  ithith  entry of a vector.

Example:  ai[l]  denotes the  ith  entry of the activations in layer  l .
We assume that you are already familiar with numpy and/or have completed the previous courses of the specialization. Let's get started!

Let's first import all the packages that you will need during this assignment. (see rnn.py)

1 - Forward propagation for the basic Recurrent Neural Network
The basic RNN that you will implement has the structure below. In this example,  Tx=TyTx=Ty .

Figure 1: Basic RNN model (refer images)

Here's how you can implement an RNN:

Steps:

Implement the calculations needed for one time-step of the RNN.
Implement a loop over  Tx  time-steps in order to process all the inputs, one at a time.
Let's go!

1.1 - RNN cell
A Recurrent neural network can be seen as the repetition of a single cell. You are first going to implement the computations for a single time-step. The following figure describes the operations for a single time-step of an RNN cell.

Figure 2: Basic RNN cell. Takes as input  x⟨t⟩  (current input) and  a⟨t−1⟩  (previous hidden state containing information from the past), and outputs  a⟨t⟩  which is given to the next RNN cell and also used to predict  y⟨t⟩ 

Exercise: Implement the RNN-cell described in Figure (2).  (refer images)

Instructions:

Compute the hidden state with tanh activation:  a⟨t⟩=tanh(Waaa⟨t−1⟩+Waxx⟨t⟩+ba) .
Using your new hidden state  a⟨t⟩ , compute the prediction  ŷ ⟨t⟩=softmax(Wyaa⟨t⟩+by) . We provided you a function: softmax.
Store  (a⟨t⟩,a⟨t−1⟩,x⟨t⟩,parameters)  in cache
Return  a⟨t⟩  ,  y⟨t⟩  and cache
We will vectorize over  mm  examples. Thus,  x⟨t⟩  will have dimension  (nx,m) , and  a⟨t⟩ will have dimension  (na,m) . (see rnn.py)

1.2 - RNN forward pass
You can see an RNN as the repetition of the cell you've just built. If your input sequence of data is carried over 10 time steps, then you will copy the RNN cell 10 times. Each cell takes as input the hidden state from the previous cell ( a⟨t−1⟩ ) and the current time-step's input data ( x⟨t⟩ ). It outputs a hidden state ( a⟨t⟩ ) and a prediction ( y⟨t⟩) for this time-step.

Figure 3: Basic RNN. The input sequence  x=(x⟨1⟩,x⟨2⟩,...,x⟨Tx⟩)  is carried over  Tx time steps. The network outputs  y=(y⟨1⟩,y⟨2⟩,...,y⟨Tx⟩) .
Exercise: Code the forward propagation of the RNN described in Figure (3). (refer images)

Instructions:

Create a vector of zeros ( a ) that will store all the hidden states computed by the RNN.
Initialize the "next" hidden state as  a0  (initial hidden state).
Start looping over each time step, your incremental index is  t  :
Update the "next" hidden state and the cache by running rnn_cell_forward
Store the "next" hidden state in  aa  ( tthtth  position)
Store the prediction in y
Add the cache to the list of caches
Return  aa ,  yy  and caches   (see rnn.py)

Congratulations! You've successfully built the forward propagation of a recurrent neural network from scratch. This will work well enough for some applications, but it suffers from vanishing gradient problems. So it works best when each output  y⟨t⟩  can be estimated using mainly "local" context (meaning information from inputs  x⟨t′⟩  where  t′ is not too far from  t ).

In the next part, you will build a more complex LSTM model, which is better at addressing vanishing gradients. The LSTM will be better able to remember a piece of information and keep it saved for many timesteps.

2 - Long Short-Term Memory (LSTM) network
This following figure shows the operations of an LSTM-cell. 

Figure 4: LSTM-cell. This tracks and updates a "cell state" or memory variable  c⟨t⟩  at every time-step, which can be different from  a⟨t⟩ .  (refer images)

Similar to the RNN example above, you will start by implementing the LSTM cell for a single time-step. Then you can iteratively call it from inside a for-loop to have it process an input with  Tx time-steps.

About the gates
- Forget gate
For the sake of this illustration, lets assume we are reading words in a piece of text, and want use an LSTM to keep track of grammatical structures, such as whether the subject is singular or plural. If the subject changes from a singular word to a plural word, we need to find a way to get rid of our previously stored memory value of the singular/plural state. In an LSTM, the forget gate lets us do this:

Γ⟨t⟩f=σ(Wf[a⟨t−1⟩,x⟨t⟩]+bf)(1)
 
Here,  WfWf  are weights that govern the forget gate's behavior. We concatenate  [a⟨t−1⟩,x⟨t⟩]  and multiply by  Wf. The equation above results in a vector  Γf⟨t⟩  with values between 0 and 1. This forget gate vector will be multiplied element-wise by the previous cell state  c⟨t−1⟩ . So if one of the values of Γf⟨t⟩  is 0 (or close to 0) then it means that the LSTM should remove that piece of information (e.g. the singular subject) in the corresponding component of  c⟨t−1⟩ . If one of the values is 1, then it will keep the information.

- Update gate
Once we forget that the subject being discussed is singular, we need to find a way to update it to reflect that the new subject is now plural. Here is the formulat for the update gate:

Γu⟨t⟩=σ(Wu[a⟨t−1⟩,x{t}]+bu)
 
Similar to the forget gate, here  Γu⟨t⟩  is again a vector of values between 0 and 1. This will be multiplied element-wise with  c̃ ⟨t⟩ , in order to compute  c⟨t⟩ .

- Updating the cell
To update the new subject we need to create a new vector of numbers that we can add to our previous cell state. The equation we use is:

c~⟨t⟩=tanh⁡(Wc[a⟨t−1⟩,x⟨t⟩]+bc)
 
Finally, the new cell state is:

c⟨t⟩=Γf⟨t⟩∗c⟨t−1⟩+Γu⟨t⟩∗c~⟨t⟩
 
- Output gate
To decide which outputs we will use, we will use the following two formulas:

Γo⟨t⟩=σ(Wo[a⟨t−1⟩,x⟨t⟩]+bo)
 
a⟨t⟩=Γo⟨t⟩∗tanh⁡(c⟨t⟩)
 
Where in equation 5 you decide what to output using a sigmoid function and in equation 6 you multiply that by the  tanh  of the previous state.

2.1 - LSTM cell
Exercise: Implement the LSTM cell described in the Figure (3).

Instructions:

Concatenate a⟨t−1⟩ and x⟨t⟩ in a single matrix: concat=[a⟨t−1⟩x⟨t⟩]
Compute all the formulas 1-6. You can use sigmoid() (provided) and np.tanh().
Compute the prediction y⟨t⟩. You can use softmax() (provided).  (see rnn.py)

2.2 - Forward pass for LSTM
Now that you have implemented one step of an LSTM, you can now iterate this over this using a for-loop to process a sequence of  Tx inputs.  
Figure 4: LSTM over multiple time-steps.    (refer images)

Exercise: Implement lstm_forward() to run an LSTM over  Tx  time-steps.

Note:  c⟨0⟩  is initialized with zeros. (see rnn.py)

Congratulations! You have now implemented the forward passes for the basic RNN and the LSTM. When using a deep learning framework, implementing the forward pass is sufficient to build systems that achieve great performance.



