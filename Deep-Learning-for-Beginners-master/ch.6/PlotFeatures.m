clear all

load('MnistConv.mat')

k  = 2;
x  = X(:, :, k);
y1 = Conv(x, W1);                 % Convolution,  20x20x20
y2 = ReLU(y1);                    %
y3 = Pool(y2);                    % Pool,         10x10x20
y4 = reshape(y3, [], 1);          %                   2000  
v5 = W5*y4;                       % ReLU,              360
y5 = ReLU(v5);                    %
v  = Wo*y5;                       % Softmax,            10
y  = Softmax(v);                  %
  

figure;
display_network(x(:));
title('Input Image')

convFilters = zeros(9*9, 20);
for i = 1:20
  filter            = W1(:, :, i);
  convFilters(:, i) = filter(:);
end
figure
display_network(convFilters);
title('Convolution Filters')

fList = zeros(20*20, 20);
for i = 1:20
  feature     = y1(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);
title('Features [Convolution]')

fList = zeros(20*20, 20);
for i = 1:20
  feature     = y2(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);
title('Features [Convolution + ReLU]')

fList = zeros(10*10, 20);
for i = 1:20
  feature     = y3(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);
title('Features [Convolution + ReLU + MeanPool]')