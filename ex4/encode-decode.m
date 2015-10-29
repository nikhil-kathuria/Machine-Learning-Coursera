clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 8;  % 20x20 Input Images of Digits
hidden_layer_size = 3;   % 25 hidden units
num_labels = 2;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)