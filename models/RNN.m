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
    properties %(Access=private)
        batchsize
        h0_
        h_ % store pre-activations
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
            initialize(rnn, inputsize, hiddensize);
        end
        function initialize(rnn, inputsize, hiddensize)
            stddev = 1/sqrt(hiddensize);
            rnn.Wh = (2*rand(hiddensize)-1)*stddev;
            rnn.Wx = (2*rand(hiddensize, inputsize)-1)*stddev;
            rnn.b = (2*randn(hiddensize,1)-1)*stddev;
        end
        
        function [h_end, h, h0] = evaluate(rnn, x, h0)
            nh = rnn.hiddensize;
            sb = cell(1, ndims(x)-2);
            [nx, sb{:}, nt] = size(x);
            nb = prod([sb{:}]);
            
            assert(nx == rnn.inputsize, 'RNN:wrongSize', 'Input size incorrect.')
            x = reshape(x, nx, nb, nt);
            
            if nargin < 3 || isempty(h0)
                h0 = zeros(nh, nb);
            end

            hx = reshape(rnn.Wx*x(:,:) + rnn.b, nh, nb, nt); % precompute Wx*x_i + b            
            h = zeros(nh, nb, nt);     % store intermediate state
            
            h_end = h0;
            for i=1:nt
                y_end = rnn.Wh*h_end + hx(:,:,i);
                h_end = rnn.activation(y_end);
                h(:,:,i) = h_end;
            end
            if ~isempty(sb)
                h_end = reshape(h_end, nh, sb{:});
            end
        end
        
        function [h_end, h] = call(rnn, x, varargin)
            % side-effect: store hidden state vs time in internal buffer
            [h_end, h, h0] = evaluate(rnn, x, varargin{:});
            rnn.h0_ = h0;
            rnn.h_ = h;
            rnn.x_ = x;
            rnn.batchsize = size(h,2);
        end
        
        function [h_end, h] = recall(rnn)
            [h_end, h] = call(rnn, rnn.x_, rnn.h0_);
        end
        
        function [h_end_v, h_v] = fdiff(rnn, v)
            % compute derivative of h w.r.t. <params, v>
            
            % retrieve buffers
            x = rnn.x_;
            h = rnn.h_;
            % compute sizes
            nh = rnn.hiddensize;
            % sb = rnn.batchsize;
            [~, nb, nt] = size(x);
            % pre-alocate ouputs
            fullseq = nargout >= 2;           
            % TODO: might want to store these to save computation time
            %       or do these in for loop instead of vectorized to save
            %       memory
            h_y = rnn.dactivation(h);   % pre-compute dactivations
            [Vh, Vx, vb] = getweights(rnn, v);

            h_end_v = Vh*rnn.h0_ + Vx*x(:,:,1) + vb;
            h_end_v = h_y(:,:,1).*h_end_v;
            if fullseq
                h_v = zeros(nh, nb, nt);
                h_v(:,:,1) = h_end_v;
            end 
            for i=2:nt
                h_end_v = rnn.Wh*h_end_v + Vh*h(:,:,i-1) + ...
                                           Vx*x(:,:,i)+...
                                           vb;
                h_end_v = h_y(:,:,i).*h_end_v;
                if fullseq
                    h_v(:,:,i) = h_end_v;
                end
            end
        end

        function [u_h_p] = bdiff(rnn, u_end, u)
            % compute derivative of <u, h> w.r.t. params
            
            % retrieve buffers
            x = rnn.x_;
            h = rnn.h_;
            % compute sizes
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            sb = rnn.batchsize;
            [~, nb, nt] = size(h); % check ~ same as nx
            fullseq = (nargin >= 3) && ~isempty(u);
            if fullseq
                u_end = u_end + u(:,:,end);
            end
            % TODO: might want to store these to save computation time
            %       or do these in for loop instead of vectorized to save
            %       memory
            h_y = rnn.dactivation(h);
            
            l_0 = (h_y(:,:,end).*u_end).';
            u_h_p = zeros(nh+nx+1, nh);
            for i=nt-1:-1:1
                u_h_p = u_h_p + [h(:,:,i); 
                                 x(:,:,i+1); 
                                 ones(1,nb)]*l_0;
                l_0 = l_0*rnn.Wh;
                if fullseq
                    l_0 = l_0 + u(:,:,i).';
                end
                l_0 = l_0.*h_y(:,:,i).';
            end
            u_h_p = u_h_p + [rnn.h0_; 
                             x(:,:,1); 
                             ones(1,nb)]*l_0;
            u_h_p = vecweights(rnn, u_h_p.'); 
        end
        
        function [u_h_p] = Bdiff(rnn, u_end, u)
            % compute derivative of <u, h> w.r.t. params
            
            % retrieve buffers
            x = rnn.x_;
            h = rnn.h_;
            % compute sizes
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            sb = rnn.batchsize;
            [~, nb, nt] = size(h); % check ~ same as nx
            fullseq = (nargin >= 3) && ~isempty(u);
            if fullseq
                u_end = u_end + u(:,:,end);
            end
            % TODO: might want to store these to save computation time
            %       or do these in for loop instead of vectorized to save
            %       memory
            h_y = rnn.dactivation(h);
            l_0 = (h_y(:,:,end).*u_end).';
            u_h_p = zeros(nh+nx+1, nh);
            for i=nt-1:-1:1
                u_h_p = u_h_p + [h(:,:,i); 
                                 x(:,:,i+1); 
                                 ones(1,nb)]*l_0;
                l_0 = l_0*rnn.Wh;
                if fullseq
                    l_0 = l_0 + u(:,:,i);
                end
                l_0 = l_0.*h_y(:,:,i).';
            end
            u_h_p = u_h_p + [rnn.h0_; 
                             x(:,:,1); 
                             ones(1,nb)]*l_0;
            u_h_p = reshape(u_h_p.', [], 1); 
        end
        
        function [Vh, Vx, vb] = getweights(rnn, v)
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            V = reshape(v, nh, []);
            Vh = V(:,1:nh);
            Vx = V(:,nh+1:nh+nx);
            vb = V(:,nh+nx+1);
        end
        function p = set_params_hook(rnn, p)
            % introduce modifications to p prior to assignment
        end
        function set.params(rnn, p)
            p = set_params_hook(rnn, p);
            [rnn.Wh, rnn.Wx, rnn.b] = getweights(rnn, p);
        end
%         function set.params(rnn, p)
%             nh = rnn.hiddensize;
%             nx = rnn.inputsize;
%             rnn.Wh(:) = p(1:nh^2);
%             rnn.Wx(:) = p(nh^2+1:nh^2+nh*nx);
%             rnn.b(:) = p(nh*(nh+nx)+1:nh*(nh+nx+1));
%         end

        function v = vecweights(rnn, V)
            if nargin == 1
                v = [rnn.Wh(:); rnn.Wx(:); rnn.b];
            else
                v = V(:);
            end
        end
        function p = get.params(rnn)
            %p = [rnn.Wh(:); rnn.Wx(:); rnn.b];
            p = vecweights(rnn);
        end
        function nh = get.hiddensize(rnn)
            nh = size(rnn.Wh, 1);
        end
        function nx = get.inputsize(rnn)
            nx = size(rnn.Wx, 2);
        end
        function np = getparamsize(rnn)
            nh = rnn.hiddensize;
            nx = rnn.inputsize;
            np = nh*(nh+nx+1);            
        end
        function np = get.paramsize(rnn)
            np = getparamsize(rnn);
        end
    end
    
    methods (Static)
        function y = relu(x)
            y = x;
            y(y<0) = 0;
        end
    end
end



