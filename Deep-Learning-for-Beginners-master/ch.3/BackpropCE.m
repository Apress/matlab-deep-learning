function [W1, W2] = BackpropCE(W1, W2, X, D)
  alpha = 0.9;
  
  N = 4;  
  for k = 1:N
    x = X(k, :)';        % x = a column vector
    d = D(k);
    
    v1 = W1*x;
    y1 = Sigmoid(v1);    
    v  = W2*y1;
    y  = Sigmoid(v);
    
    e     = d - y;
    delta = e;

    e1     = W2'*delta;
    delta1 = y1.*(1-y1).*e1; 
    
    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;
    
    dW2 = alpha*delta*y1';    
    W2 = W2 + dW2;
  end
end