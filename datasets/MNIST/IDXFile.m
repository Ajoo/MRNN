classdef IDXFile < handle
    %IDXFILE Summary of this class goes here
    %   Detailed explanation goes here
    properties
        fid
        size
        type
        elementsize
        origin
        outputtype
    end
    
    methods (Static)
        function [type, size] = getelementtype(code)
            switch code
                case 8
                    type = 'uchar';
                    size = 1;
                case 9
                    type = 'schar';
                    size = 1;
                case 11
                    type = 'int16';
                    size = 2;
                case 12
                    type = 'int32';
                    size = 4;
                case 13
                    type = 'float';
                    size = 4;
                case 14
                    type = 'double';
                    size = 8;
            end
        end
    end
    
    methods
        function h = IDXFile(filename, outputprecision)
            %IDXFILE Construct an instance of this class
            %   Detailed explanation goes here
            fid = fopen(filename, 'r', 'b');
            
            mn = fread(fid, 4, 'uchar', 0, 'b');
            assert(mn(1)==0 && mn(2)==0, 'Wrong Header: First two bytes are not zero!');
            
            [type, elementsize] = IDXFile.getelementtype(mn(3));
            dims = mn(4);
            
            h.origin = (dims + 1)*4;
            h.size = fread(fid, dims, 'int32', 0, 'b')';
            h.elementsize = elementsize;
            h.type = type;
            h.fid = fid;
            if nargin >= 2
                if outputprecision == '*'
                    h.outputtype = ['*' type];
                else
                    h.outputtype = [type '=>' outputprecision];
                end
            else
                h.outputtype = type;
            end
        end
        function delete(h)
            fclose(h.fid);
        end
        function data = read(h, size, start, origin)
            % start is 1-based
            nelements = prod(h.size(2:end));
            if nargin < 2
                size = h.size(1);
                start = 1;
                origin = -1;
            end
            if nargin >= 3 && ~isempty(start)
                if nargin < 4 || isempty(origin)
                    origin = -1;
                end
                offset = (start-1)*nelements*h.elementsize;
                if (isnumeric(origin) && origin == -1) || strcmpi(origin, 'bof')
                    offset = offset + h.origin;
                end
                fseek(h.fid, offset, origin);
            end
            
            data = fread(h.fid, [nelements, size], h.outputtype, 0, 'b');
        end
        function reset(h)
            fseek(h.fid, h.origin, -1);
        end
    end
end

