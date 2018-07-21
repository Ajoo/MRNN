function [x, y] = samplecopy(imax, ncopy, nb, nt)
%SAMPLECOPY Summary of this function goes here
%   Detailed explanation goes here
    seq = randi(imax, 1, nb, ncopy);
    
    x = cat(3, seq, zeros(1, nb, nt+ncopy));
    x(:,:,nt+ncopy) = imax+1;
    y = cat(3, zeros(1, nb, nt+ncopy), seq);
end

