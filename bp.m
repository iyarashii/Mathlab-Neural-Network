clear all;      % clear workspace
nntwarn off;    % turn off nnt warnings
format long     % use long display format
format compact  % use compact display format

load wino_norm  % load normalized data

[R,Q] = size(P);    % Ustalenie rozmiaru wejscia
[S3,Q] = size(T);   % define number of neurons in output layer
P = Pn;             % assign normalized P values to P variable

S1 = 10; 		    % number of neurons in the input layer
S2 = 5;      		% number of neurons in the hidden layer

[W1,B1] = nwtan(S1,R);		% neuron initialization in each layer
[W2,B2] = nwtan(S2,S1);		% and the initialization of their weight matrix and bias matrix
[W3,B3] = rands(S3,S2);

disp_freq = 100;      		% display frequency
max_epoch = 5000;    		% max epoch count
err_goal = 0.25;     		% error goal
lr = 1e-4;  				% learning rate

error = [];        			% error storing table




for epoch=1:max_epoch,		% loop executing learning algorithm for current step
   
    A1 = tansig(W1*P,B1);		% Calculating value on the first layer output
    A2 = tansig(W2*A1,B2);		% Calculating value on the second layer output
    A3 = purelin(W3*A2,B3);		% Calculating value on the third layer output
	
    E = T - A3;					% Calculating difference between current output value and target output value

    D3 = deltalin(A3,E);		% Calculating delta for new weights for each layer
    D2 = deltatan(A2,D3,W3);
    D1 = deltatan(A1,D2,W2);   
    
    

    [dW1,dB1] = learnbp(P,D1,lr);	% calculating new weight using error backpropagation method
    W1 = W1 + dW1;					% correction for first layer
    B1 = B1 + dB1;
    [dW2,dB2] = learnbp(A1,D2,lr);
    W2 = W2 + dW2;					% correction for second layer 
    B2 = B2 + dB2;
    [dW3,dB3] = learnbp(A2,D3,lr);
    W3 = W3 + dW3;					% correction for third layer
    B3 = B3 + dB3;
 
    SSE = sumsqr(E);				% sum squared elements of matrix E
    error = [error SSE];			% add SSE to the table after each iteration
   
   
    if SSE < err_goal,				% end learning process if error goal has been reached
        epoch = epoch - 1;
        break,
    end,
	
    if isnan(SSE),          % interrupt learning process if SSE reached NaN(not a number) value
      epoch = epoch - 1;
      break,
	end,
   

    if(rem(epoch,disp_freq)==0)		% projection condition, if the remainder = 0
        epoch						%  print variable values during the projection
        SSE
	   % graph
       plot(1:length(A3),A3,'b',1:length(T),T,'r');
       title(['Epoch: ' int2str(epoch)]);
       legend('Learned values','Real values',4);
       pause(1e-100)
    end 
end
%   print variables after learning process has successfully ended
[T' A3' (T-A3)' (abs(T-A3)>=0.5)']	% display current calculated values and target values
epoch
SSE
procent = 100*(1-sum((abs(T-A3)>=.5)')/length(T))	% calculate and display network learn percentage
