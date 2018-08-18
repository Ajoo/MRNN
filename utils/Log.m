classdef Log < handle
    %LOGGER Wrapper class around a structure of arrays
    
    properties
        chunksize = 10e3;
    end
    
    properties
        data
        names
        idx = 0
    end
    
    methods
        function log = Log(varargin)
            %LOGGER Construct an instance of this class
            %   Detailed explanation goes here
            if nargin==1 && isnumeric(varargin{1})
                ncol = varargin{1};
            else
                ncol = numel(varargin);
            end
            log.data = zeros(log.chunksize, ncol);
            log.names = varargin;
        end
        
        function append(log, varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            if log.idx >= length(log)
                log.data = [log.data; zeros(log.chunksize, ncols(log))];
            end
            log.idx = log.idx + 1;
            log.data(log.idx, :) = [varargin{:}];
        end
        
        function f = getfield(log, name)
            col = contains(log.names, name);
            f = log.data(1:log.idx, col);
        end
        
        function fn = fieldnames(log)
            fn = log.names;
        end
        
        function d = getdata(log)
            d = log.data(1:log.idx, :);
        end
        
        function l = length(log)
            l = size(log.data, 1);
        end
        
        function n = ncols(log)
            n = size(log.data, 2);
        end
    end
end

