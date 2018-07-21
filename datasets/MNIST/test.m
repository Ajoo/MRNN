fid = fopen('train-images-idx3-ubyte', 'r');
%%
frewind(fid)
mn = fread(fid, 4, 'uchar', 0, 'b');
%%
mn = fread(fid, 1, 'uint32', 0, 'b');
sizes = fread(fid, 3, 'uint32', 0, 'b');
%%
mn = fread(fid, 1, 'uint32', 0, 'b');
ni = fread(fid, 1, 'uint32', 0, 'b');
nh = fread(fid, 1, 'uint32', 0, 'b');
nw = fread(fid, 1, 'uint32', 0, 'b');

%%
fclose(fid);