function [x, y] = samplecopy(imax, ncopy, nb, nt)
%SAMPLECOPY Summary of this function goes here
%   Detailed explanation goes here
    seq = randi(imax, nb, ncopy);
    
    x = cat(2, seq, zeros(nb, nt+nb));
    x(:,nb+nt) = imax+1;
    y = cat(2, zeros(nb, nt+ncopy), seq);
end

