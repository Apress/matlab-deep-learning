function W = DeltaSGD(W, X, D)
  alpha = 0.9;
  
  [N,M]=size(X);

  for k = 1:N
    x = X(k, :)';
    d = D(k);

    v = W*x;
    y = Sigmoid(v);
    
    e     = d - y;  
    delta = y*(1-y)*e;
  
    dW = alpha*delta*x;     % delta rule  

    for i =1:M
        W(i) = W(i) + dW(i); 
    end
  end
end