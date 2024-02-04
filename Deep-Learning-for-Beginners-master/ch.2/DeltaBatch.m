function W = DeltaBatch(W, X, D)
  alpha = 0.9;

  [N,M]=size(X);

  dWsum = zeros(M, 1);

  for k = 1:N
    x = X(k, :)';
    d = D(k);
                        
    v = W*x;
    y = Sigmoid(v);
  
    e     = d - y;    
    delta = y*(1-y)*e;
    
    dW = alpha*delta*x;
    
    dWsum = dWsum + dW;
  end
  dWavg = dWsum / N;
  
  for i =1:M
    W(i) = W(i) + dWavg(i);
  end
end
