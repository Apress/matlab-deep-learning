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
      
W = 2*rand(1, 3) - 1;

for epoch = 1:40000           % train
  W = DeltaXOR(W, X, D);
end

N = 4;                        % inference
for k = 1:N
  x = X(k, :)';
  v = W*x;
  y = Sigmoid(v)
end