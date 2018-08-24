function [x, y] = truncate_seq(sample, I)
%TRUNCATE_BATCH Summary of this function goes here
%   Detailed explanation goes here
    [x, y] = sample();
    x = x(:,:,I);
    if ndims(y) == 3
        y = y(:,:,I);
    end
end

