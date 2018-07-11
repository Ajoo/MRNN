function [x, y] = sampleaddition(B, T, Tmax)
%ADDITIONBATCH Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 3
        Tmax = floor(T/2);
    end
    x = cat(1, rand(1, B, T), zeros(1, B, T));
    idx = choose2(B, Tmax);
    
    x(2, idx(:,1)) = 1;
    x(2, idx(:,2)) = 1;
    y = x(1, idx(:,1)) + x(1, idx(:,2));
end

function idx = choose2(B, Tmax)
    idx = randi(Tmax-1, B, 2);
    idx(2, idx(:,1)==idx(:,2)) = Tmax;
    idx = sub2ind([B, Tmax], [1:B; 1:B].', idx);
end