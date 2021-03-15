function [blocked_img] = add_block2img(img,blocksize,ratio)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[a b]=size(img);
n=a/blocksize;
m=b/blocksize;
blocked_img=zeros(a,b);
for i=1:1:n
    for j=1:1:m
        sumblock=sum(sum(img((i-1)*blocksize+1:1:i*blocksize,(j-1)*blocksize+1:1:j*blocksize)));
        if sumblock/(blocksize^2)>ratio
            blocked_img((i-1)*blocksize+1:1:i*blocksize,(j-1)*blocksize+1:1:j*blocksize)=1;
        end
            
    end
end
imagesc(blocked_img)
end

