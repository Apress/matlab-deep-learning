clear

X = [ 0 0 1;
      0 1 1;
      1 0 1;
      1 1 1;
    ];

D = [ 0
      0
      1
      1
    ];

[N,M]=size(X);  
W = 2*rand(1, M) - 1;

for epoch = 1:40000 
  W = DeltaBatch(W, X, D); 
end

for k = 1:N
  x = X(k, :)';
  v = W*x;
  y = Sigmoid(v);
  disp([' y_',num2str(k),' = ', num2str(y)])
end

