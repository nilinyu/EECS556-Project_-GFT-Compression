clear all
t = Tiff('4.2.07.tiff');
Img=rgb2gray(read(t));
imagesc(Img);
img=Img;%8X8 BLOCK
[Ix Iy]=gradient(double(img));
Ix2=Ix.^2;
Iy2=Iy.^2;
Ixy=Ix.*Iy;
[a b]=size(img);
clear check
check=[];
num=0;
imgg=zeros(a,b);
for i=1:1:a
   for j=1:1:b 
       st=[Ix2(i,j) Ixy(i,j);Ixy(i,j) Iy2(i,j)];   
       det_img=det(st);                          
       trace_img=trace(st);                        
       check=[check; det_img trace_img];
       i
       j
   end
end

%class one: smooth blocks
s=find(ismember([abs(check(:,1))>=0 check(:,2)<150], [1 1],'rows')==1);
im=ones(size(img))*255;
im(s)=0;im=im';
[blocked_img] = add_block2img(im>0,16,0.12);
Imgg=Img;
Imgg(find(blocked_img==1))=0;
imagesc(Imgg)
imst=blocked_img;

%%  the other half: Img
%class three & class two: cornor
s=find(ismember([abs(check(:,1))>0.2e-10 check(:,2)>150], [1 1],'rows')==1);
im=ones(size(img))*0;
im(s)=255;im=im';
[blocked_img] = add_block2img(im>0,16,0.0);
Imgg=Img;
Imgg(find(blocked_img==0))=0;
imagesc(Imgg)

Imgg=Img;
ims3=(imst-blocked_img)>=1;
Imgg(find(ims3==0))=0;
imagesc(Imgg)


imgf=imresize(imgg,size(Img));
[
figure;
plot(check(:,1),check(:,2),'.');
ylabel('trace');xlabel('det');



figure;
imshow(imgg,[])
