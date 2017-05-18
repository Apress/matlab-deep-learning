clear all
           
X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      1
      1
      0
    ];
      
W1 = 2*rand(4, 3) - 1;
W2 = 2*rand(1, 4) - 1;

for epoch = 1:10000           % train
  [W1 W2] = BackpropMmt(W1, W2, X, D);
end

N = 4;                        % inference
for k = 1:N
  x  = X(k, :)';
  v1 = W1*x;
  y1 = Sigmoid(v1);
  v  = W2*y1;
  y  = Sigmoid(v)
end