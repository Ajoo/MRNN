classdef RNN < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Wh
        Wx
        b
        activation
        dactivation
    end
    properties (Access=private)
        batchsize
        h0_
        y_ % store pre-activations
        x_ % store inputs
    end
    properties (Dependent)
        params
        inputsize
        hiddensize
        paramsize
    end
    
    methods
        function rnn = RNN(inputsize, hiddensize, activation)
            if nargin < 3
                activation = 'tanh';
            end
            stddev = 1/sqrt(hiddensize);
            rnn.Wh = (2*rand(hiddensize)-1)*stddev;
            rnn.Wx = (2*rand(hiddensize, inputsize)-1)*stddev;
            rnn.b = (2*randn(hiddensize,1)-1)*stddev;
            switch upper(activation)
                case 'TANH'
                    rnn.activation = @tanh;
                    rnn.dactivation = @(z) 1-z.^2;
                case 'RELU'
                    rnn.activation = @RNN.relu;
                    rnn.dactivation = @(z) z>=0;
                case 'LINEAR'
                    rnn.activation = @(x) x;
                    rnn.dactivation = @(z) ones(size(z));
            end
        end
        
        function [h_end, h] = recall(rnn)
            [h_end, h] = call(rnn, rnn.x_, rnn.h0_);
        end
        
        function [h_end, h] = call(rnn, x, h0)
            fullseq = nargout >= 2;
            
            % side-effect: store hidden state vs time in internal buffer
            nh = rnn.hiddensize;
            sb = cell(1, ndims(x)-2);
            [nx, sb{:}, nt] = size(x);
            nb = prod([sb{:}]);
            
            assert(nx == rnn.inputsize, 'RNN:wrongSize', 'Input size incorrect.')
            x = reshape(x, nx, nb, nt);
            
            if nargin < 3
                h0 = zeros(nh, nb);
            end

            hx = reshape(rnn.Wx*x(:,:) + rnn.b, nh, nb, nt); % precompute Wx*x_i + b            
            
            if fullseq
                h = zeros(nh, nb, nt);     % store intermediate state
            end
            y = zeros(nh, nb, nt);
            h_end = h0;
            for i=1:nt
                y_end = rnn.Wh*h_end + hx(:,:,i);
                h_end = rnn.activation(y_end);
                y(:,:,i) = y_end;
                if fullseq
                    h(:,:,i) = h_end;
                end
            end
            if ~isempty(sb)
                h_end = reshape(h_end, nh, sb{:});
            end
            rnn.h0_ = h0;
            rnn.y_ = y;
            rnn.x_ = x;
            rnn.batchsize = sb;
        end
        
        function [h_end_v, h_v] = fdiff(rnn, v)
            % compute derivative of h w.r.t. <params, v>
            
            % retrieve buffers
            x = rnn.x_;
            y = rnn.y_;
            % compute sizes
            nh = rnn.hiddensize;
            sb = rnn.batchsize;
            [nx, nb, nt] = size(y);
            % pre-alocate ouputs
            fullseq = nargout >= 2;
            if fullseq
                h_v = zeros(nx, nb, nt);
            end
            h_end_v = zeros(nx, nb);
            
            % TODO: might want to rework this to save memory...
            hprev = cat(3, rnn.h0_, rnn.activation(y));
            hx1 = [hprev(:,:,1:end-1); x; ones(1,nb,nt)];    % re-compute hidden-states
            h_y = rnn.dactivation(hprev(:,:,2:end));         % pre-compute dactivations
            v = reshape(v, nh, []);
            
            for i=1:nt
                h_end_v = rnn.Wh*h_end_v + v*hx1(:,:,i);
                h_end_v = h_y(:,:,i).*h_end_v;
                if fullseq
                    h_v(:,:,i) = h_end_v;
                end
            end
        end
        % TODO: handle case where u is only size [nh, nb] assuming 
        % u(:,:,i) = 0 for i ~= nt
        function [u_h_p] = bdiff(rnn, u_end)
            % compute derivative of <u, h> w.r.t. params
            
            % retrieve buffers
            x = rnn.x_;
            y = rnn.y_;
            % compute sizes
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            sb = rnn.batchsize;
            [~, nb, nt] = size(y); % check ~ same as nx
            np = rnn.paramsize;
            % TODO: might want to rework this to save memory...
            hprev = cat(3, rnn.h0_, rnn.activation(y));
            hx1 = [hprev(:,:,1:end-1); x; ones(1,nb,nt)];    % re-compute hidden-states
            h_y = rnn.dactivation(hprev(:,:,2:end));         % pre-compute dactivations
            
            % pre-alocate ouputs
            l_0 = h_y(:,:,end).*u_end;
            u_h_p = l_0*hx1(:,:,end)';
            for i=nt-1:-1:1
                l_0 = rnn.Wh'*l_0;
                l_0 = h_y(:,:,i).*l_0;
                u_h_p = u_h_p + l_0*hx1(:,:,i)';
            end
            
            u_h_p = u_h_p(:);
        end
        
%         function [u_h_p] = bdiff(rnn, u)
%             compute derivative of <u, h> w.r.t. params
%             
%             retrieve buffers
%             x = rnn.x_;
%             y = rnn.y_;
%             compute sizes
%             nh = rnn.hiddensize;
%             nx = rnn.inputsize;
%             sb = rnn.batchsize;
%             [~, nb, nt] = size(y); % check ~ same as nx
%             np = rnn.paramsize;
%             pre-alocate ouputs
%             l_0 = zeros(nh, nb);
%             u_h_p = zeros(nh, nh+nx+1);
%             TODO: might want to rework this to save memory...
%             hprev = cat(3, rnn.h0_, rnn.activation(y));
%             hx1 = [hprev(:,:,1:end-1); x; ones(1,nb,nt)];    % re-compute hidden-states
%             h_y = rnn.dactivation(hprev(:,:,2:end));         % pre-compute dactivations
%             for i=nt:-1:1
%                 l_0 = rnn.Wh'*l_0 + u(:,:,i);
%                 l_0 = h_y(:,:,i).*l_0;
%                 u_h_p = u_h_p + l_0*hx1(:,:,i)';
%             end
%             
%             u_h_p = u_h_p(:);
%         end
        
        function set.params(rnn, p)
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            rnn.Wh(:) = p(1:nh^2);
            rnn.Wx(:) = p(nh^2+1:nh^2+nh*nx);
            rnn.b(:) = p(nh*(nh+nx)+1:nh*(nh+nx+1));
        end
        function p = get.params(rnn)
            p = [rnn.Wh(:); rnn.Wx(:); rnn.b];
        end
        function nh = get.hiddensize(rnn)
            nh = size(rnn.Wh, 1);
        end
        function nx = get.inputsize(rnn)
            nx = size(rnn.Wx, 2);
        end
        function np = get.paramsize(rnn)
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            np = nh*(nh+nx+1);
        end
    end
    
    methods (Static)
        function y = relu(x)
            y = x;
            y(y<0) = 0;
        end
    end
end


