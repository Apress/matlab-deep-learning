function y = Softmax(x)
  ex = exp(x);
  y  = ex / sum(ex);
end